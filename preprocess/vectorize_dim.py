import numpy
import argparse
import sys

tag_matrix = {"no":[1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
              "lo":[0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
              "fn":[0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
              "gr":[0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
              "to":[0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
              "so":[0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
              "ao":[0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
              "st":[0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
              "e":[0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
              "x":[0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
              "kk":[0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
              "kvk":[0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
              "hk":[0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
              "ókyngr":[0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
              "et":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
              "ft":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
              "nf":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
              "þf":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
              "þgf":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
              "ef":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
              "vsk_gr":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
              "sérn":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
              "sb":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
              "vb":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
              "ób":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
              "fst":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
              "mst":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
              "est":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
              "ábfn":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
              "óákv_ábfn":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
              "efn":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
              "óákv_fn":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
              "pfn":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
              "sfn":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
              "tfn":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
              "1p":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
              "2p":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
              "3p":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
              "frumt":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
              "árt":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
              "prós":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
              "fjöldat":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
              "nh":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
              "bh":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
              "fh":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
              "vh":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
              "sagnb":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
              "lhn":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
              "lhþ":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
              "gm":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
              "mm":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
              "nt":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0],
              "þt":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0],
              "ekki_fallst":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0],
              "upphr":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0],
              "st_þol":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0],
              "st_þag":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0],
              "st_ef":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0],
              "nhm":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0],
              "tilvt":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0],
              "stýfður":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0],
              "afn": [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0],
              "óp": [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0],
              "spurnar": [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0],
              "sérst": [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]}

def build_tagarray(current):
    temp = ''
    #cases
    if current.startswith('NF'):
        temp = numpy.array(tag_matrix['nf'])
        current = current[2:]
    elif current.startswith('ÞGF'):
        temp = numpy.array(tag_matrix['þgf'])
        current = current[3:]
    elif current.startswith('ÞF'):
        temp = numpy.array(tag_matrix['þf'])
        current = current[2:]
    elif current.startswith('EF'):
        temp = numpy.array(tag_matrix['ef'])
        current = current[2:]
    elif current.startswith('-NF'):
        temp = numpy.array(tag_matrix['nf'])
        current = current[3:]
    elif current.startswith('-ÞF'):
        temp = numpy.array(tag_matrix['þf'])
        current = current[3:]
    elif current.startswith('-ÞGF'):
        temp = numpy.array(tag_matrix['þgf'])
        current = current[4:]
    elif current.startswith('-EF'):
        temp = numpy.array(tag_matrix['ef'])
        current = current[3:]
    elif current.startswith('_NF'):
        temp = numpy.array(tag_matrix['nf'])
        current = current[3:]
    elif current.startswith('_ÞF'):
        temp = numpy.array(tag_matrix['þf'])
        current = current[3:]
    elif current.startswith('_ÞGF'):
        temp = numpy.array(tag_matrix['þgf'])
        current = current[4:]
    elif current.startswith('_EF'):
        temp = numpy.array(tag_matrix['ef'])
        current = current[3:]

    #aðeins óákveðin fornöfn eru sérstæð
    elif current.startswith('-SERST'):
        temp = numpy.array(tag_matrix['sérst'])
        current = current[6:]

    #number
    elif current.startswith('ET'):
        temp = numpy.array(tag_matrix['et'])
        current = current[2:]
    elif current.startswith('FT'):
        temp = numpy.array(tag_matrix['ft'])
        current = current[2:]
    elif current.startswith('-ET'):
        temp = numpy.array(tag_matrix['et'])
        current = current[3:]
    elif current.startswith('-FT'):
        temp = numpy.array(tag_matrix['ft'])
        current = current[3:]

    #person
    elif current.startswith('1P'):
        temp = numpy.array(tag_matrix['1p'])
        current = current[2:]
    elif current.startswith('2P'):
        temp = numpy.array(tag_matrix['2p'])
        current = current[2:]
    elif current.startswith('3P'):
        temp = numpy.array(tag_matrix['3p'])
        current = current[2:]
    elif current.startswith('-1P'):
        temp = numpy.array(tag_matrix['1p'])
        current = current[3:]
    elif current.startswith('-2P'):
        temp = numpy.array(tag_matrix['2p'])
        current = current[3:]
    elif current.startswith('-3P'):
        temp = numpy.array(tag_matrix['3p'])
        current = current[3:]

    #article
    elif current.startswith('gr'):
        temp = numpy.array(tag_matrix['vsk_gr'])
        current = current[2:]

    elif current.startswith('LHÞT'):
        temp = numpy.array(tag_matrix['lhþ'])
        current = current[4:]
    elif current.startswith('-VB'):
        temp = numpy.array(tag_matrix['vb'])
        current = current[3:]
    elif current.startswith('-SB'):
        temp = numpy.array(tag_matrix['sb'])
        current = current[3:]
    elif current.startswith('OP'):
        temp = numpy.array(tag_matrix['óp'])
        current = current[2:]
    elif current.startswith('-það'):
        temp = numpy.array(tag_matrix['óp'])
        current = current[4:]
    elif current.startswith('LHNT'):
        temp = numpy.array(tag_matrix['lhn'])
        current = current[4:]
    elif current.startswith('LH-NT'):
        temp = numpy.array(tag_matrix['lhn'])
        current = current[5:]
    elif current.startswith('SP'):
        temp = numpy.array(tag_matrix['spurnar'])
        current = current[2:]

    #gender
    elif current.startswith('-KK'):
        temp = numpy.array(tag_matrix['kk'])
        current = current[3:]
    elif current.startswith('-KVK'):
        temp = numpy.array(tag_matrix['kvk'])
        current = current[4:]
    elif current.startswith('-HK'):
        temp = numpy.array(tag_matrix['hk'])
        current = current[3:]
    elif current.startswith('KK'):
        temp = numpy.array(tag_matrix['kk'])
        current = current[2:]
    elif current.startswith('KVK'):
        temp = numpy.array(tag_matrix['kvk'])
        current = current[3:]
    elif current.startswith('HK'):
        temp = numpy.array(tag_matrix['hk'])
        current = current[2:]

    #voice
    elif current.startswith('MM'):
        temp = numpy.array(tag_matrix['mm'])
        current = current[2:]
    elif current.startswith('-MM'):
        temp = numpy.array(tag_matrix['mm'])
        current = current[3:]
    elif current.startswith('GM'):
        temp = numpy.array(tag_matrix['gm'])
        current = current[2:]
    elif current.startswith('-GM'):
        temp = numpy.array(tag_matrix['gm'])
        current = current[3:]

    #mood
    elif current.startswith('-NH'):
        temp = numpy.array(tag_matrix['nh'])
        current = current[3:]
    elif current.startswith('-FH'):
        temp = numpy.array(tag_matrix['fh'])
        current = current[3:]
    elif current.startswith('-VH'):
        temp = numpy.array(tag_matrix['vh'])
        current = current[3:]
    elif current.startswith('-BH'):
        temp = numpy.array(tag_matrix['bh'])
        current = current[3:]
    elif current.startswith('-SAGNB'):
        temp = numpy.array(tag_matrix['sagnb'])
        current = current[6:]
    elif current.startswith('-ST'):
        temp = numpy.array(tag_matrix['stýfður'])
        current = current[3:]

    #tense
    elif current.startswith('-NT'):
        temp = numpy.array(tag_matrix['nt'])
        current = current[3:]
    elif current.startswith('-ÞT'):
        temp = numpy.array(tag_matrix['þt'])
        current = current[3:]

    elif current.startswith('FSB'):
        temp = numpy.array(tag_matrix['fst'])
        temp += numpy.array(tag_matrix['sb'])
        current = current[3:]
    elif current.startswith('FVB'):
        temp = numpy.array(tag_matrix['fst'])
        temp += numpy.array(tag_matrix['vb'])
        current = current[3:]
    elif current.startswith('ESB'):
        temp = numpy.array(tag_matrix['est'])
        temp += numpy.array(tag_matrix['sb'])
        current = current[3:]
    elif current.startswith('EVB'):
        temp = numpy.array(tag_matrix['est'])
        temp += numpy.array(tag_matrix['vb'])
        current = current[3:]
    elif current.startswith('FST'):
        temp = numpy.array(tag_matrix['fst'])
        current = current[3:]
    elif current.startswith('MSTSB'):
        temp = numpy.array(tag_matrix['mst'])
        temp += numpy.array(tag_matrix['sb'])
        current = current[5:]
    elif current.startswith('MST2'):
        temp = numpy.array(tag_matrix['mst'])
        current = current[4:]
    elif current.startswith('MST'):
        temp = numpy.array(tag_matrix['mst'])
        current = current[3:]
    elif current.startswith('EST'):
        temp = numpy.array(tag_matrix['est'])
        current = current[3:]
    elif current.startswith('OBEYGJANLEGT'):
        temp = numpy.array(tag_matrix['ób'])
        current = current[12:]
    return current, temp

def vectorise_all(word_form_list, outfile):
    bin_dict = {}
    ctr = 0
    wfl_length = len(word_form_list)
    for i in word_form_list:
        ctr += 1
        if ctr % 10000 == 0:
            print(str(ctr) + ' of ' + str(wfl_length))
        current_2 = i.split(';')[2].strip()
        current_3 = i.split(';')[3].strip()
        current = i.split(';')[5].strip().strip('2').strip('3')
        current_wordform = i.split(';')[4].strip()

        # creating the vectors - do this more properly
        if current_2 == 'kk':
            temp = numpy.array(tag_matrix['no']) + numpy.array(tag_matrix['kk'])
        if current_2 == 'kvk':
            temp = numpy.array(tag_matrix['no']) + numpy.array(tag_matrix['kvk'])
        if current_2 == 'hk':
            temp = numpy.array(tag_matrix['no']) + numpy.array(tag_matrix['hk'])
        if current_2 == 'so':
            temp = numpy.array(tag_matrix['so'])
        if current_2 == 'lo':
            temp = numpy.array(tag_matrix['lo'])
        if current_2 == 'to':
            temp = numpy.array(tag_matrix['to'])
        if current_2 == 'gr':
            temp = numpy.array(tag_matrix['gr'])
        if current_2 == 'ao':
            temp = numpy.array(tag_matrix['ao'])
        if current_2 == 'fn':
            temp = numpy.array(tag_matrix['fn'])
        if current_2 == 'rt': #add to tag_matrix?
            temp = numpy.array(tag_matrix['lo'])
        if current_2 == 'pfn':
            temp = numpy.array(tag_matrix['fn']) + numpy.array(tag_matrix['pfn'])
        if current_2 == 'fs':
            temp += numpy.array(tag_matrix['st_þag']) + numpy.array(tag_matrix['st_þol']) + numpy.array(tag_matrix['st_ef'])
        if current_2 == 'st':
            temp += numpy.array(tag_matrix['st'])
        if current_2 == 'nhm':
            temp += numpy.array(tag_matrix['nhm'])
        if current_2 == 'uh':
            temp += numpy.array(tag_matrix['upphr'])
        if current_2 == 'afn':
            temp += numpy.array(tag_matrix['afn'])

        # In the latest version of DIM there may be more categories of proper nouns
        if current_3 in ['heö','fyr','örn','föð','ism','móð','gæl','lönd','erl','göt','hetja','mvirk','bær','þor','hug','erm','dýr','ætt']:
            temp += numpy.array(tag_matrix['sérn'])

        while len(current) > 0:
            current_out, mark = build_tagarray(current)
            if current_out == current:
                print(i, current)
                current = ''
            else:
                temp += numpy.array(mark)
                current = current_out

        #using a dict for it all - merging all possibilities for a wordform into one vector
        if current_wordform in bin_dict:
            bin_dict[current_wordform] = numpy.logical_or(bin_dict[current_wordform], temp)
        else:
            bin_dict[current_wordform] = temp

    with open(outfile, "w") as f:
        for j in bin_dict.keys():
            if len(j) > 0:
                if len(bin_dict[j]) > 0:
                    try:
                        f.write(j + ';' + numpy.array2string(1 * numpy.array(bin_dict[j]), max_line_width=200,
                                                             separator=',') + '\n')
                    except:
                        print(numpy.array2string(eval(bin_dict[j])))
                        sys.exit(0)


if __name__ == '__main__':
    # reading input parameters
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input', '-i', help='Name of input file.', default="./data/SHsnid.csv")
    parser.add_argument('--output', '-o', help='Name of output file.', default="./extra/dmii.vectors")

    try:
        args = parser.parse_args()
    except:
        sys.exit(0)

    dim_file = open(args.input, 'r')
    wordforms = dim_file.readlines()
    vectorise_all(wordforms, args.output)
