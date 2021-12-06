import argparse
import pickletools

p = argparse.ArgumentParser(description="check protocol of a pickle file")
p.add_argument("path", help="path to the pickle file")
opts = p.parse_args()

with open(opts.path, "rb") as fin:
    pops = pickletools.genops(fin)
    proto = (
        2 if next(pops)[0].proto == 2 else int(any(op.proto for op, fst, snd in pops))
    )
    print("Pickle protocol:", proto)
