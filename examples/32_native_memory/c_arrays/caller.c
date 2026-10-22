#include <assert.h>
#include <stdint.h>
#include <stdio.h>

#include "array_api.h"

_Static_assert(_Generic((int *)0, int32_t *: 1, default: 0),
               "this example requires C int to be int32_t");

int main(void) {
    int ice[8][8] = {{0}};
    game_init();
    process(ice);
    for (int row = 0; row < 8; ++row)
        for (int col = 0; col < 8; ++col)
            assert(ice[row][col] == row * 8 + col);
    puts("original C array updated: ice[7][7] = 63");
    return 0;
}
