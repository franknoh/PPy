//! The same three questions put to Rust: `i64` panics on overflow in debug, wraps in release.

fn may_overflow(n: i64) -> i64 {
    let mut result: i64 = 1;
    for i in 1..=n {
        result *= i;
    }
    result
}

fn floor_semantics(a: i64, b: i64) -> i64 {
    a / b
}

fn modulo_semantics(a: i64, b: i64) -> i64 {
    a % b
}

fn main() {
    println!("{}", may_overflow(20));
    println!("{}", may_overflow(30));
    println!("{} {}", floor_semantics(-7, 2), floor_semantics(7, 2));
    println!("{} {}", modulo_semantics(-7, 2), modulo_semantics(7, -2));
}
