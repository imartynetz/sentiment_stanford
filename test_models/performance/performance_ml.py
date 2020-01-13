import pandas as pd
import numpy as np
import common_modules
import pdpipe as pdp
from log_reg_perf import reg_log
from nayve_bayes_perf import naive_bayes


def main():
    train, test = common_modules.merge_files()

    pipeline = pdp.ApplyByCols("texto", common_modules.tag_remove, "clean_texto", drop=False)
    pipeline += pdp.ApplyByCols("clean_texto", common_modules.trat_texto)
    train = pipeline(train)
    test = pipeline(test)
    reg_log(train.texto, train.label, test.texto, test.label)
    naive_bayes(train.texto, train.label, test.texto, test.label)


if __name__ == "__main__":
    main()
