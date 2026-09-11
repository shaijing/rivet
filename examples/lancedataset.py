import io

import lance
from PIL import Image

root = "/data/datasets/custom/nuimages"

train = lance.dataset(f"{root}/train.lance")
val = lance.dataset(f"{root}/val.lance")

print("Train rows:", train.count_rows())
print("Val rows:", val.count_rows())

print("\nTrain schema:")
print(train.schema)


# 随机读取第 1000 个样本
index = 1000

table = train.take([index])
row = table.to_pylist()[0]

print("\nSample:")
print("label:", row["label"])

# 如果你保存的是:
# image: binary
# label: int32
encoded = row["image"]

print("encoded bytes:", len(encoded))
print("signature:", encoded[:8])

# 解码图片
image = Image.open(io.BytesIO(encoded))

print("format:", image.format)
print("size:", image.size)
print("mode:", image.mode)

image.show()
