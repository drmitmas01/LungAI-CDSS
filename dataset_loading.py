import pylidc as pl

scans = pl.query(pl.Scan).all()

print("Number of scans found:", len(scans))