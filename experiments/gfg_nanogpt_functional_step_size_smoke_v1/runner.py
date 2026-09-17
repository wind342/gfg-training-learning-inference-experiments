"""Run selection and adjudication in distinct processes."""
import subprocess
import sys

from .common import parser


def main():
    p = parser()
    p.add_argument("--stage", choices=("all", "select", "adjudicate"), default="all")
    args = p.parse_args()
    if args.stage == "all":
        options = []
        for name in ("reports_root", "response_root", "boundary_root", "output_root"):
            options += ["--" + name.replace("_", "-"), str(getattr(args, name).resolve())]
        for stage in ("select", "adjudicate"):
            subprocess.run([sys.executable, "-B", "-m", __spec__.name, "--stage", stage, *options], check=True)
    elif args.stage == "select":
        from .selector import run
        run(args)
    else:
        from .adjudicate import run
        run(args)


if __name__ == "__main__":
    main()
