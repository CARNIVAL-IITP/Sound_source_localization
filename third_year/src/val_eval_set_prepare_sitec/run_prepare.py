from prepare_main import val_csv_prepare, eval_csv_prepare
import sys


if __name__=='__main__':
    

    t=val_csv_prepare(sys.argv[1], 'val')

    t=eval_csv_prepare(sys.argv[1], 'test')

    