# alpha191
gtja alpha 191 python implementation

## input
`Dict<key, pd.DataFrame>`
The key includes the required data, such as `close, open, high, low, volume, etc`.

The key's value is a wide table dataframe, where columns are stocks, rows are dates. (Faster than some `groupby` methods.)

## output
A wide table dataframe of the alpha values.