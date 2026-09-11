import lance

ds = lance.dataset("/tmp/data.lance")

print("=== Schema ===")
print(ds.schema)

print("\n=== Rows ===")
print(ds.count_rows())

print("\n=== Version ===")
print(ds.version)

table = ds.to_table()
print(table)