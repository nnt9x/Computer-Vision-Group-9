
TRAIN_POS_LST = 'dataset/Train/pos.lst'
TRAIN_POS_DIR = 'dataset/Train/'

pos_lines = None
with open(TRAIN_POS_LST) as f:
    pos_lines = f.readlines()

print(pos_lines)
print(type(pos_lines))
