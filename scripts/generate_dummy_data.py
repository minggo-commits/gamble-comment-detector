import pandas as pd

judi_examples = [
    Main slot gacor sini, deposit cepat lewat WA 0812-xxx",
    "Daftar judi online, bonus new member, klik link",
    "Promo bandar togel hari ini, pasang angka via WA",
    "Situs sbobet terbaik, cashback besar, daftar sekarang",
    "Join agen poker online, withdrawal cepat"
]

normal_examples = [
    "Barang sesuai deskripsi, terima kasih penjual",
    "Dimana toko ini berada? ada diskon akhir pekan?",
    "Bagus kualitasnya, pengiriman cepat",
    "Apakah tersedia ukuran L? kapan restock?",
    "Terima kasih untuk pelayanannya"
]

rows = []
for i in range(1000):
    if i % 5 == 0:
        rows.append({"text": judi_examples[i % len(judi_examples)], "label": "judi"})
    else:
        rows.append({"text": normal_examples[i % len(normal_examples)], "label": "normal"})

df = pd.DataFrame(rows)
df.to_csv("data/comments.csv", index=False)
print("Dummy data generated and saved to data/comments.csv", len(df), "rows")