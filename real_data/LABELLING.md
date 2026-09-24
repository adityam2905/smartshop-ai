# Labelling real listings

`labels.csv` has one row per real Google Shopping listing. Fill in two columns
per row, judging each listing **on its own merits** — don't try to guess what
the model will do (the sheet deliberately doesn't show it).

Open it in Excel or Google Sheets, fill `legit`, `good_deal` and optionally
`notes`, and save it back as CSV with the same name.

## `legit` — is this a genuine seller and a genuine product?

| Value | When |
|---|---|
| `1` | A real business selling the real product. You'd be comfortable ordering. |
| `0` | A scam, counterfeit/replica ("first copy", "7A", "master copy"), or a seller you wouldn't trust with your money. |
| *blank* | You can't tell after a quick check — the row is left out of the results. |

How to check (a minute per unfamiliar seller; the same seller repeats a lot):
- Known retailer (Flipkart, Amazon, Croma, Reliance Digital, the brand's own
  store…) → `1`.
- Unfamiliar seller → search its name + "reviews", or look it up on
  Trustpilot / ScamAdviser. An established site with a real history → `1`;
  warnings, a days-old domain, or no footprint at all → `0` or blank.
- The title or price says replica / first copy / "AAA quality" → `0`, even
  from a real shop — it isn't the genuine product.

## `good_deal` — is this price at least a little cheaper than usual?

Only needed when `legit` is `1` (leave it blank for `0` rows).

| Value | When |
|---|---|
| `1` | Cheaper than the typical price for **this exact variant** (storage, size, colour, new vs refurbished) at major retailers. |
| `0` | At the typical price or above it, or overpriced (e.g. an import at a big markup). |

How to check: compare with the same variant on Amazon/Flipkart/the brand's
store. Refurbished or used items count as a good deal only if they're cheaper
than *other refurbished* units, not just cheaper than new.

## Rows you can skip

- **No seller** (e.g. "₹29,000+"): a Google price-comparison card, not a
  listing from one shop. The evaluation leaves these out automatically.
- **Off-topic results** (a steam iron for "Philips air fryer") are still worth
  labelling — the app shows them, so how it handles them counts.

## Tips

- Be consistent rather than perfect. If unsure, leave it blank — blanks are
  excluded, guesses add noise.
- Use `notes` for anything odd ("price is for the case, not the phone").
- ~150 rows takes about 45–60 minutes; sellers repeat, so it speeds up.

When done: `python -m experiments.evaluate_real`
