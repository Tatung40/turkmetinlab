"""Checks real public TES-Lex Excel; prints only schema and aggregate results."""
from tmlab import teslex
df = teslex.load_source()
print("TES_LEX_ROW_COUNT:",len(df))
print("TES_LEX_COLUMNS:",list(df.columns))
print("TES_LEX_SAMPLE_SCHEMA:",[(str(col), list(df[col].astype(str).head(2))) for col in df.columns])
assert len(df) >= 18000, "Expected >18k lexicon entries"
print("GERCEK_TESLEX_EXCEL_YUKLEME_BASARILI")
