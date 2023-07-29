"""Demo script for teneva_bm.

Call it as "python demo.py" to run demo for all existing benchmarks or as
"python demo.py bm_<name_of_collection>_<name_of_benchmark>" to run an example
for a specific benchmark "<name_of_benchmark>" from the collection (folder)
"<name_of_collection" (e.g., "python demo.py bm_qubo_knap_det"). Note that the
code for the demo run is at the end of the corresponding benchmark file in the
section "if __name__ == '__main__':".

You can also run this script as ""python demo.py info" to present only short
info about all existing collections and benchmarks.

"""
import sys


from teneva_bm import teneva_bm_demo


if __name__ == '__main__':
    bm_use = sys.argv[1] if len(sys.argv) > 1 else None

    only_info = bm_use and bm_use.lower() == 'info'
    bm_use = None if only_info else bm_use

    teneva_bm_demo(bm_use, with_info=(bm_use is None), all=(not only_info))
