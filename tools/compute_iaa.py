from stutter.utils import Vector, fragment_by_overlaps

def prepare_data(df, gold=None):
    # construct timevr
    df['timevr'] = df['time'].apply(lambda x: Vector(x[0], x[1]))

    # select cols
    cols = ['annotator','media_file', 'label', 'timevr']
    df = df[cols]

    # fragment by overlaps
    grannodf = fragment_by_overlaps(df, 'annotator', 'media_file', 'label', decomp_fn=unionize_vectorrange_sequence, gold_df=gold)

    # merge mean
    return grannodf

def main():

    