"""`tensor-fusion`: a chain of elementwise tensor operations becomes one loop.

Every elementwise operation, lowered on its own, is a loop that writes a
whole tensor another loop then reads: one temporary and one pass over
memory per operation. Where an operation's only reader is another
elementwise operation in the same block, the two are one computation per
element, and this pass says so: the maximal such tree -- `unary`, the
binary arithmetic, `fill`, `broadcast`, and a `reduce` at the root --
becomes a single `tensor.fused` whose body is the scalar arithmetic over
one element of each input:

    %r = tensor.fused %a, %b, %s : tensor.tensor<f64, N> {
    ^body(%x: f64, %y: f64, %z: f64):
        %t = math.sqrt %x : f64
        %u = core.mul %t, %y : f64
        %v = core.add %u, %z : f64
        tensor.yield %v
    }

`lower-tensor` makes one loop of it, with no temporary between the
operations; a reduce at the root folds each element straight into the
accumulator. The pass is what the directive's fusion analysis is at this
level; a GPU or XLA backend reads the same region.
"""

from __future__ import annotations

from .. import shape as shapes
from ..dialects import tensor as tensors
from ..dialects.tensor_patterns import canonicalization_patterns
from ..model import Block, Builder, IRFunction, IRModule, Operation, Value
from ..passes import FunctionPass, PassContext
from ..pattern import GreedyRewriteDriver, PatternSet, Rewriter

__all__ = ["FUSIBLE", "FuseTensor", "TensorCanonicalize"]

#: What joins a fused region: elementwise arithmetic, and the views whose
#: element is the operand's element.
FUSIBLE = frozenset(
    {
        "tensor.unary",
        "tensor.fill",
        "tensor.broadcast",
        *(f"tensor.{n}" for n in tensors.ELEMENTWISE),
    }
)
#: What may stand at the root of a group on its own.
_ROOTS = FUSIBLE - {"tensor.fill", "tensor.broadcast"}


class TensorCanonicalize(FunctionPass):
    """The tensor dialect's own patterns, on their own (`canonicalize` includes them)."""

    name = "tensor-canonicalize"
    preserves = ("dominators",)

    def run_on_function(self, function: IRFunction, ctx: PassContext) -> bool:
        rewriter = Rewriter()
        changes = GreedyRewriteDriver(PatternSet(canonicalization_patterns())).run(
            function, rewriter
        )
        for line in rewriter.log:
            ctx.remark(f"@{function.name}: {line}")
        return changes > 0


class FuseTensor(FunctionPass):
    name = "tensor-fusion"
    #: Operations are replaced within their blocks; no branch changes.
    preserves = ("dominators",)

    def __init__(self) -> None:
        super().__init__()
        self.module: IRModule | None = None

    def run(self, module: IRModule, ctx: PassContext) -> bool:
        self.module = module
        return super().run(module, ctx)

    def run_on_function(self, function: IRFunction, ctx: PassContext) -> bool:
        fused = 0
        for block in list(function.body.blocks):
            for group in _plan(block):
                names = ", ".join(dict.fromkeys(_spell(op) for op in group))
                self._fuse(group)
                fused += 1
                ctx.remark(
                    f"@{function.name}: tensor ops fused: {names} ({len(group)} operations) "
                    "into one loop"
                )
        return fused > 0

    def _fuse(self, group: list[Operation]) -> None:
        """Replace `group` -- in block order, the root last -- with one `tensor.fused`."""
        root = group[-1]
        members = {id(op) for op in group}
        inputs: list[Value] = []
        seen: set[int] = set()
        for op in group:
            for operand in op.operands:
                owner = operand.owner
                if isinstance(owner, Operation) and id(owner) in members:
                    continue
                if id(operand) not in seen:
                    seen.add(id(operand))
                    inputs.append(operand)
        attributes: dict = {}
        if root.name == "tensor.reduce":
            attributes = {
                "reduce": root.attributes["op"],
                "axes": tuple(root.attributes["axes"]),  # type: ignore[arg-type]
                "keepdims": bool(root.attributes.get("keepdims", False)),
            }
        b = Builder().before(root)
        fused = b.create("tensor.fused", tuple(inputs), (root.results[0].type,), attributes)
        body = fused.add_region().add_block("body", [(None, _element(v)) for v in inputs])
        mapping: dict[int, Value] = {
            id(value): argument for value, argument in zip(inputs, body.arguments, strict=True)
        }
        bb = Builder(body)
        produced: Value | None = None
        for op in group:
            if op is root and op.name == "tensor.reduce":
                produced = mapping[id(op.operands[0])]
                break
            operands = [mapping.get(id(v), v) for v in op.operands]
            if op.name in {"tensor.fill", "tensor.broadcast"}:
                value = operands[0]
            elif op.name == "tensor.unary":
                value = tensors.scalar_unary(bb, str(op.attributes["op"]), operands[0], self.module)
            else:
                value = tensors.scalar_binary(
                    bb, op.local_name, operands[0], operands[1], self.module
                )
            mapping[id(op.results[0])] = value
            produced = value
        assert produced is not None
        tensors.yield_(bb, produced)
        root.results[0].replace_all_uses_with(fused.results[0])
        for op in reversed(group):
            op.erase()


def _shape_from_inputs(members: list[Operation], member_ids: set[int]) -> bool:
    """Do the group's tensor inputs alone give the shape the body runs over?

    A `fill` inside the group is a scalar to the fused operation, so a
    shape only a fill supplied -- `x + fill c` where `c` is wider than `x`,
    or a reduce of nothing but fills -- would be lost. Such a group stays
    as it is.
    """
    root = members[-1]
    wanted = root.operands[0].type if root.name == "tensor.reduce" else root.results[0].type
    expected = tensors.describe(wanted)
    assert expected is not None
    shape: shapes.Shape | None = None
    for op in members:
        for operand in op.operands:
            owner = operand.owner
            if isinstance(owner, Operation) and id(owner) in member_ids:
                continue
            info = tensors.describe(operand.type)
            if info is None:
                continue
            try:
                shape = info.shape if shape is None else shapes.broadcast(shape, info.shape)
            except shapes.ShapeError:
                return False
    return shape == expected.shape


def _spell(op: Operation) -> str:
    return str(op.attributes["op"]) if op.name == "tensor.unary" else op.local_name


def _element(value: Value):  # type: ignore[no-untyped-def]
    info = tensors.describe(value.type)
    return value.type if info is None else info.dtype


def _plan(block: Block) -> list[list[Operation]]:
    """The groups to fuse in `block`, each in block order with its root last.

    Walking backwards, every unclaimed root pulls in the fusible producers
    whose only reader it is; a group of one is left as it stands.
    """
    order = {id(op): index for index, op in enumerate(block.operations)}
    claimed: set[int] = set()
    groups: list[list[Operation]] = []
    roots = _ROOTS | {"tensor.reduce"}
    for root in reversed(block.operations):
        if id(root) in claimed or root.name not in roots:
            continue
        members = [root]
        member_ids = {id(root)}
        pending = [root]
        while pending:
            consumer = pending.pop()
            for operand in consumer.operands:
                producer = operand.owner
                if (
                    isinstance(producer, Operation)
                    and producer.name in FUSIBLE
                    and producer.parent is block
                    and id(producer) not in claimed
                    and id(producer) not in member_ids
                    # Every reader of the producer is already in the group.
                    and all(id(user) in member_ids for user, _index in operand.uses)
                ):
                    members.append(producer)
                    member_ids.add(id(producer))
                    pending.append(producer)
        if len(members) < 2:
            continue
        members.sort(key=lambda op: order[id(op)])
        if not _shape_from_inputs(members, member_ids):
            continue
        claimed.update(member_ids)
        groups.append(members)
    # Groups were found root-first from the end; fuse them in block order.
    groups.reverse()
    return groups
