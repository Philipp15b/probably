from probably.pgcl.compiler import parse_pgcl
from probably.prism.backend import translate_to_prism


def test_ber_ert():
    program = parse_pgcl("""
        nat x;
        nat n;
        nat r;
        while (x < n) {
            r := 1 : 1/2 + 0 : 1/2;
            x := x + r;
            tick(1);
        }
    """)
    translate_to_prism(program)


def test_linear01():
    program = parse_pgcl("""
        nat x;
        while (2 <= x) {
            { x := x - 1; } [1/3] {
                x := x - 2;
            }
            tick(1);
        }
    """)
    translate_to_prism(program)


def test_prspeed():
    program = parse_pgcl("""
        nat x;
        nat y;
        nat m;
        nat n;
        while ((x + 3 <= n)) {
            if (y < m) {
                { y := y + 1; } [1/2] {
                    y := y + 0;
                }
            } else {
                { x := x + 0; } [1/4] {
                    { x := x + 1; } [1/3] {
                        { x := x + 2; } [1/2] {
                            x := x + 3;
                        }
                    }
                }
            }
            tick(1);
        }
    """)
    translate_to_prism(program)
