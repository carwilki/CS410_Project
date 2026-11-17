# Draft evaluation queries — notes

All 30 queries were written by hand; no LLM was used to generate them. I read about 20 short narratives from one parquet row group to get a feel for the vocabulary and copied none of them. Output is in `draft.tsv`. Length buckets: short ≤12 words, medium 13–28, long >28. The draft has 11 short, 12 medium and 7 long.

## Product-family grouping

The corpus has 3,851,415 rows. CFPB renamed products in 2017 and again in 2023, so the 21 raw product labels collapse into 10 families:

| Family | Raw products (years) |
|---|---|
| Credit reporting | Credit reporting (2015–17); Credit reporting, credit repair services, or other personal consumer reports (2017–23); Credit reporting or other personal consumer reports (2023–26) |
| Debt collection | Debt collection |
| Credit card | Credit card; Credit card or prepaid card (2017–23) |
| Bank account | Checking or savings account (2017–26); Bank account or service (2015–17) |
| Mortgage | Mortgage |
| Prepaid / money transfer | Money transfer, virtual currency, or money service; Money transfers; Virtual currency; Prepaid card; Other financial service |
| Student loan | Student loan |
| Vehicle loan | Vehicle loan or lease; Consumer Loan (2015–17; assumed mostly vehicle, sub_product not checked) |
| Payday/personal loan | Payday loan; Payday loan, title loan, or personal loan; Payday loan, title loan, personal loan, or advance loan |
| Debt/credit management | Debt or credit management (2023–26) |

Two caveats about this grouping:
- "Credit card or prepaid card" (2017–23) mixes credit and prepaid cards. I put it under Credit card, so that family is slightly inflated.
- Issue labels were renamed too, for example "Problem with a credit reporting company's investigation…" became "Problem with a company's investigation…". I treat the old and new names as one issue.

## Distribution and allocation

| Family | n | Share | Proportional (×30) | Allocated |
|---|---:|---:|---:|---:|
| Credit reporting | 2,511,515 | 65.2% | 19.6 | 16 |
| Debt collection | 445,246 | 11.6% | 3.5 | 4 |
| Credit card | 237,827 | 6.2% | 1.9 | 2 |
| Bank account | 204,233 | 5.3% | 1.6 | 2 |
| Mortgage | 147,249 | 3.8% | 1.1 | 2 |
| Prepaid / money transfer | 135,594 | 3.5% | 1.1 | 1 |
| Student loan | 63,196 | 1.6% | 0.5 | 1 |
| Vehicle loan | 63,121 | 1.6% | 0.5 | 1 |
| Payday/personal loan | 37,828 | 0.98% | 0.3 | 1 |
| Debt/credit management | 5,606 | 0.15% | 0.05 | 0 |

How the allocation was set:
- Every family with at least 1% of rows gets at least one query.
- Payday/personal loan sits just under the cutoff at 0.98%. I gave it one query anyway because it is a distinct lending type.
- Credit reporting was cut from about 19–20 queries (strict proportional) to 16 (53%). This frees slots so Mortgage gets a second query and Debt collection rounds up to 4, while credit reporting stays clearly dominant. For a strictly proportional set, use CR 18–19 / DC 3 / CC 2 / Bank 1–2 / Mortgage 1 / Money transfer 1 / SL 1 / VL 1 / PD 1.
- Debt/credit management (0.15%) gets no query.

## Top issues per family (share within the family)

- **Credit reporting:** Incorrect information 47.2% (sub-issue "belongs to someone else" alone is 694k, 28% of CR); Improper use 26.3% ("used improperly" 507k, "unrecognized inquiries" 148k); Investigation problem 23.1% (both names combined; "did not fix error" 368k, ">30 days" 109k); fraud alerts/freezes 0.6%.
- **Debt collection:** Debt not owed 43.0% (not yours 100k, identity theft 57k, already paid 28k); Written notification 20.1%; False statements 11.2%; Threatened legal action 8.5%; Communication tactics 7.7%.
- **Credit card:** Purchase on statement 23.2%; Other 10.7%; Fees or interest 9.9%; Getting a card 9.6%.
- **Bank account:** Managing an account 50.5% (deposits/withdrawals 36k, debit/ATM card 20k); Closing 12.8%; Unauthorized charges 11.6%; Low funds/overdraft 9.2%.
- **Mortgage:** Trouble during payment process 38.6%; Struggling to pay 18.6%; Applying/refinancing 10.2%; legacy servicing/escrow 10.0%.
- **Prepaid / money transfer:** Other transaction problem 40.5%; Fraud or scam 20.4%; Unauthorized transactions 7.3%.
- **Student loan:** Dealing with lender/servicer 68.3% (both names combined; payment handling 12k, bad info 12k); Struggling to repay 16.2%.
- **Vehicle loan:** Managing the loan 34.8% (billing 8.5k); End of loan 14.8% (title after payoff 4k); Getting a loan 11.5%; Repossession 11.3%.
- **Payday/personal loan:** Unexpected fees or interest 24.4%; Payment problems 12.2%; Struggling to pay 11.5%.

## Query-to-issue map

| Family | Issue | Queries |
|---|---|---|
| Credit reporting | Information belongs to someone else | q01–q04 |
| Credit reporting | Account status incorrect | q05 |
| Credit reporting | Account information incorrect | q06 |
| Credit reporting | Personal information incorrect | q07 |
| Credit reporting | Old information reappears | q08 |
| Credit reporting | Improper use | q09, q11 |
| Credit reporting | Unrecognized inquiries | q10, q12 |
| Credit reporting | Investigation did not fix error | q13 |
| Credit reporting | Not notified of investigation status | q14 |
| Credit reporting | Investigation took more than 30 days | q15 |
| Credit reporting | Fraud alert / security freeze | q16 |
| Debt collection | Debt is not yours | q17 |
| Debt collection | Debt was paid | q18 |
| Debt collection | Not enough information to verify debt | q19 |
| Debt collection | Frequent calls plus legal threat (two issues) | q20 |
| Credit card | Purchase dispute not resolved | q21 |
| Credit card | Fees | q22 |
| Bank account | Deposits and withdrawals | q23 |
| Bank account | Overdraft fees | q24 |
| Mortgage | Escrow | q25 |
| Mortgage | Loan modification / foreclosure | q26 |
| Prepaid / money transfer | Payment-app scam | q27 |
| Student loan | Payment handling | q28 |
| Vehicle loan | Title after payoff | q29 |
| Payday/personal loan | Unexpected interest | q30 |

q11 imitates the FCRA template letters common in the corpus. They are a large share of credit-reporting narratives, so one query in that register is useful.

## Corpus observations relevant to evaluation

- **Credit reporting dominates.** It is 65% of all narratives. All but 31,587 of those (1.3%) fall under the post-2017 product names.
- **Template text is reused across products.** Credit-repair template narratives often appear verbatim under several products. For example, the same "I disputed the account and asked for proof of charge off…" text is filed under Credit card and Payday loan; near-identical "company I never had any dealings with" text appears under Debt collection. Product labels on retrieved documents will therefore be noisy, and many hits will be near-duplicates. Consider deduplicating, or judging relevance on the text rather than the label.
- **Some issue fields are empty.** Mortgage, Money transfer and Payday loan issues often have no sub-issue.

## Script (run from the project root with `uv run python dist.py`)

```python
import pandas as pd
df = pd.read_parquet("artifacts/corpus.parquet", columns=["product","issue","sub_issue","date"])
N = len(df); print("rows", N)
print(df["product"].value_counts().to_string())
FAM = {
 "Credit reporting": "Credit reporting",
 "Credit reporting, credit repair services, or other personal consumer reports": "Credit reporting",
 "Credit reporting or other personal consumer reports": "Credit reporting",
 "Debt collection": "Debt collection",
 "Mortgage": "Mortgage",
 "Credit card": "Credit card",
 "Credit card or prepaid card": "Credit card",
 "Prepaid card": "Prepaid / money transfer",
 "Checking or savings account": "Bank account",
 "Bank account or service": "Bank account",
 "Money transfer, virtual currency, or money service": "Prepaid / money transfer",
 "Money transfers": "Prepaid / money transfer",
 "Virtual currency": "Prepaid / money transfer",
 "Other financial service": "Prepaid / money transfer",
 "Student loan": "Student loan",
 "Vehicle loan or lease": "Vehicle loan",
 "Consumer Loan": "Vehicle loan",
 "Payday loan, title loan, or personal loan": "Payday/personal loan",
 "Payday loan, title loan, personal loan, or advance loan": "Payday/personal loan",
 "Payday loan": "Payday/personal loan",
 "Debt or credit management": "Debt/credit management",
}
df["family"] = df["product"].map(FAM).fillna(df["product"])
fam = df["family"].value_counts()
print("\nFAMILY"); print(pd.DataFrame({"n":fam,"share":(fam/N).round(4)}).to_string())
d = pd.to_datetime(df["date"], errors="coerce")
print("\nproduct year range"); print(df.assign(y=d.dt.year).groupby("product")["y"].agg(["min","max"]).to_string())
for f in fam.index:
    s = df[df.family==f]
    print(f"\n== {f} ({len(s)})")
    vc = s["issue"].value_counts().head(6)
    print(pd.DataFrame({"n":vc,"share":(vc/len(s)).round(3)}).to_string())
    vc2 = (s["issue"].fillna("")+" | "+s["sub_issue"].fillna("")).value_counts().head(5)
    print(vc2.to_string())
```

Sub-issue detail was produced with the same data, filtered by product and grouped on `issue | sub_issue` with `value_counts().head(N)`. The calibration narratives came from `pq.ParquetFile(...).read_row_group(mid, columns=['product','narrative'])`, keeping narratives of 150–350 characters, two per product.
