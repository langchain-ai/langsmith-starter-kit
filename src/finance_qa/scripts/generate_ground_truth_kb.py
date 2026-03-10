#!/usr/bin/env python3
"""Generate the Finance QA knowledge base (data/traces/ground_truth_kb.csv).

Writes synthetic Q&A rows covering 9 credit card support topics.
Run from the starter-kit root:

    python src/finance_qa/scripts/generate_ground_truth_kb.py
"""
import csv
import json
from pathlib import Path

_OUTPUT = Path(__file__).parent.parent / "data" / "traces" / "ground_truth_kb.csv"

TOPICS = {
    "payment_processing": {
        "chunks": [
            """Payment Processing: Overview
Payments can be made through multiple channels including online banking, mobile app, phone, mail, or in-person at branch locations
Payments must be received by 5:00 PM ET on the due date to be considered on-time
Minimum payment amount is calculated as either $35 or 1% of balance plus interest and fees, whichever is greater
Remember: Payments take 1-3 business days to process depending on payment method
Tell customer:
If paying online: Payment will post within 1 business day if made before 8:00 PM ET
If paying by phone: Payment will post within 1 business day
If paying by mail: Allow 5-7 business days for processing
If paying at branch: Payment posts same business day""",
            """Automatic Payment Setup
Customers can set up automatic payments in three ways: Full Balance, Minimum Payment, or Fixed Amount
Access Online Banking and navigate to Account Services > Automatic Payments
Select payment frequency: Monthly on due date or bi-weekly
Verify bank account information is correct before enabling
Automatic payments will begin on the next billing cycle after setup
Important: Customer must maintain sufficient funds in linked account
If automatic payment fails due to insufficient funds, customer will be charged returned payment fee
Customer should still monitor statements even with automatic payments enabled
Changes to automatic payment settings take effect on next billing cycle""",
            """Payment Methods Available
Online: Through chase.com or mobile app - processes within 1 business day
Phone: Call automated system or speak with representative - processes within 1 business day
Mail: Send check or money order to payment address on statement - allow 5-7 days
Branch: Visit any Chase branch location - posts same business day
Wire Transfer: For large payments or expedited processing - contact customer service
Important Notes:
Do not send cash through mail
Include account number on check memo line
Keep confirmation number for all electronic payments
Payments made after due date may incur late fees even if in transit""",
        ],
        "answer": "Payments can be made through online banking, mobile app, phone, mail, or branch locations and must be received by 5:00 PM ET on the due date to be considered on-time. Customers can set up automatic payments for full balance, minimum payment, or fixed amount. Payment processing times vary: online and phone payments post within 1 business day, mail payments take 5-7 days, and branch payments post same business day. Automatic payments require maintaining sufficient funds in the linked account to avoid returned payment fees.",
    },
    "disputes_chargebacks": {
        "chunks": [
            """Dispute Process: Filing a Dispute
Customer has 60 days from statement date to dispute a charge
Disputes can be filed online, by phone, or in writing
Provisional credit may be issued within 10 business days depending on dispute type
Required information for dispute:
- Transaction date and amount
- Merchant name
- Reason for dispute
- Any supporting documentation
Types of disputes:
- Unauthorized transactions: Charges customer didn't make or authorize
- Billing errors: Charged wrong amount or duplicate charges
- Service disputes: Goods/services not received or not as described
Tell customer: Disputed amount will be temporarily credited while investigation is in progress
Investigation typically takes 30-90 days depending on complexity""",
            """Unauthorized Transaction Disputes
If customer reports unauthorized transactions, immediately place fraud hold on account
Ask customer:
- Do they still have physical card in possession?
- When did they last use the card?
- Do they recognize the merchant or transaction?
- Has anyone else had access to card information?
For confirmed fraud:
- Issue provisional credit within 10 business days
- Cancel current card and issue replacement
- File fraud report with appropriate authorities
- Customer is not liable for unauthorized charges
Important: Customer must sign affidavit of fraud for unauthorized transaction claims""",
            """Chargeback Rights and Timeline
Under Fair Credit Billing Act, customers have protections for billing errors
Customer must notify bank in writing within 60 days of statement showing error
Bank has 30 days to acknowledge receipt of dispute
Bank must resolve dispute within 90 days (or two billing cycles)
During investigation:
- Cannot report account as delinquent to credit bureaus for disputed amount
- Cannot charge interest on disputed amount
If dispute is found in customer's favor: permanent credit will be issued
If dispute is found in merchant's favor: provisional credit will be reversed
Customer has right to appeal decision within 10 days of notification""",
        ],
        "answer": "Customers have 60 days from the statement date to dispute a charge and can file disputes online, by phone, or in writing. Provisional credit may be issued within 10 business days. The investigation typically takes 30-90 days. Under the Fair Credit Billing Act, the bank must resolve disputes within 90 days and cannot report the disputed amount as delinquent during the investigation.",
    },
    "rewards_programs": {
        "chunks": [
            """Rewards Program Overview
Standard earning rates: 1 point per $1 (base), 2 points per $1 on dining/travel (preferred), 3 points per $1 on select categories (premium), 5 points per $1 on bonus categories
Points post within 1-2 billing cycles after purchase
Points never expire as long as account remains open and in good standing
Minimum redemption amount is 2,500 points ($25 value)""",
            """Rewards Redemption Options
Cash Back: statement credit (1-2 billing cycles), direct deposit (5-7 days), or check (10-14 days)
Travel: book through rewards portal or reimburse travel purchases
Gift Cards: 150+ retailers; electronic cards delivered instantly, physical cards in 7-10 days
Merchandise: free shipping, ships within 2-3 weeks
Important: Redemptions cannot be reversed once processed""",
            """Combining and Transferring Points
Points can be combined between eligible cards on same account
Both cards must be enrolled in same rewards program; transfer is instant
Points cannot be transferred to different customers or after account closure (must redeem within 30 days)
Business and personal card points cannot be combined""",
        ],
        "answer": "Rewards points are earned on eligible purchases with rates ranging from 1 to 5 points per dollar. Points can be redeemed for cash back, travel, gift cards, or merchandise and never expire while the account is open. Minimum redemption is 2,500 points. Points can be combined between eligible cards on the same account.",
    },
    "card_activation": {
        "chunks": [
            """Card Activation: Getting Started
Activation channels: mobile app (instant), online banking, phone (automated or rep), ATM
Required: last 4 digits, expiration date, CVV, customer ID
Card activates immediately after successful verification
Old card is automatically deactivated when new card is activated""",
            """Activation Deadlines
Cards should be activated within 30 days of receipt
If not activated within 45 days, card is automatically deactivated for security
Replacement cards for expired cards activate automatically on expiration date
Virtual card numbers can be generated instantly through mobile app""",
            """Troubleshooting Activation Issues
Card won't activate online/app: clear cache, try different browser, verify credentials
Automated phone fails: use keypad instead of voice, call during business hours
Security hold preventing activation: recent fraud alert or address verification needed — contact customer service
Important: Never activate a card received unexpectedly""",
        ],
        "answer": "New cards must be activated within 30 days of receipt via the mobile app, online banking, phone, or ATM. Activation is instant after successful verification and the old card is automatically deactivated. Cards not activated within 45 days are deactivated for security. Activation issues are often caused by incorrect card info, browser cache, or security holds.",
    },
    "fraud_protection": {
        "chunks": [
            """Fraud Monitoring and Detection
Account monitored 24/7; automated systems analyze purchase patterns, locations, amounts, and high-risk merchants
Alerts sent via text, email, mobile app push notification, or phone call
Customer should respond: text YES if legitimate, NO if fraudulent
Bank will never ask for full card number or PIN in alert messages""",
            """Zero Liability Protection
Customers not responsible for unauthorized transactions on card-present, card-not-present, and contactless transactions
Requirements: report promptly, take reasonable care of card, don't share PIN
Exceptions: authorized user transactions, transactions after failing to report lost/stolen card, PIN-based transactions if PIN was compromised
Provisional credit issued within 10 business days""",
            """Reporting Lost or Stolen Cards
Report immediately via mobile app, online banking, or phone (24/7)
After reporting: card immediately deactivated, replacement card ordered, expedited shipping available (2-3 business days), virtual card number issued instantly
Customer is not liable for fraudulent charges after reporting""",
        ],
        "answer": "Accounts are monitored 24/7 for suspicious activity. Under zero liability protection, customers are not responsible for unauthorized transactions if reported promptly. Lost or stolen cards should be reported immediately through the mobile app, online banking, or phone. Provisional credit is issued within 10 business days and a replacement card is ordered automatically.",
    },
    "balance_transfers": {
        "chunks": [
            """Balance Transfer: Overview
Transfer debt from other credit cards to consolidate balances and take advantage of promotional APR
Balance transfer fee: typically 3% or 5% (minimum $5)
Processing time: 7-14 business days
Cannot transfer balances between Chase accounts""",
            """How to Request Balance Transfer
Request via online banking, phone, mobile app, or balance transfer checks
Required: account number of card to be paid off, name on account, transfer amount, account holder address
Transfer request is not a guarantee of approval; subject to available credit and account standing""",
            """Balance Transfer Terms
Promotional APR starts when transfer posts; missing a payment may void the promotional rate
Balance transfer fee is included in total balance and cannot be waived
New purchases may accrue interest at standard APR during promotional period""",
        ],
        "answer": "Balance transfers allow moving debt from other credit cards to take advantage of promotional APR. The transfer fee is typically 3-5% (minimum $5) and transfers take 7-14 business days. The promotional APR starts when the transfer posts and requires on-time payments to maintain. Transfers cannot be made between Chase accounts.",
    },
    "credit_limit_increase": {
        "chunks": [
            """Requesting a Credit Limit Increase
Channels: online banking, mobile app, or phone
Eligibility: account open 6+ months, in good standing, no recent increase (within 6 months), current income required
Factors: payment history, credit utilization, debt-to-income ratio, recent inquiries
Decision typically immediate for online requests; some require 2-7 business days""",
            """Credit Limit Decisions
Approved: new limit effective immediately, no fee
Denied: written explanation within 7-10 business days; can reapply after 6 months
Common denial reasons: insufficient income, recent late payments, high DTI, too many inquiries
Automatic increases: bank may increase periodically for accounts in good standing without hard inquiry""",
            """Credit Limit Decreases
Bank may reduce limit due to payment delinquency, decreased credit score, or account inactivity
Customer notified before effective date; decrease cannot put account over limit
Can request reconsideration if decrease seems to be an error""",
        ],
        "answer": "Credit limit increases can be requested via online banking, mobile app, or phone after 6 months with no late payments. The bank considers payment history, credit utilization, and income. Decisions are typically immediate online. The bank may also automatically increase limits for accounts in good standing without a hard inquiry.",
    },
    "statement_inquiries": {
        "chunks": [
            """Accessing Statements
Online banking: up to 7 years, downloadable PDF, available 24 hours after cycle closes
Mobile app: current and past statements
Paper statements: mailed 2-3 days after cycle closes; may incur monthly fee""",
            """Statement Cycle and Due Date
Cycle: 28-31 days; payment due date is 21-25 days after statement date
Must be received by 5:00 PM ET to be on-time
Grace period: no interest on new purchases if previous balance was paid in full""",
            """Common Statement Issues
Missing statement: check spam, verify email/address, request duplicate online
Incorrect information: dispute within 60 days
Change delivery method or due date (once per year): takes 1-2 billing cycles to take effect""",
        ],
        "answer": "Statements are available through online banking (up to 7 years), mobile app, or by mail. Electronic statements are available 24 hours after the cycle closes. The payment due date is 21-25 days after the statement date. Due dates can be changed once per year and take 1-2 billing cycles to take effect.",
    },
    "interest_fees": {
        "chunks": [
            """Understanding APR
APR types: purchase, balance transfer, cash advance, and penalty APR
Most cards have variable APR tied to the Prime Rate
Interest can be avoided by paying the full balance by the due date each month""",
            """Interest Calculation
Method: Average Daily Balance — sum daily balances, divide by days in cycle, multiply by daily periodic rate (APR / 365)
Cash advances accrue interest immediately with no grace period""",
            """Common Fees and Waivers
Late payment: $29 (first), $40 (subsequent within 6 months)
Returned payment: $29
Foreign transaction: 3% (some cards exempt)
Waiver: one-time courtesy waiver likely for first occurrence with good standing history""",
        ],
        "answer": "APR is the yearly borrowing cost and varies by transaction type. Interest is calculated using the average daily balance method and can be avoided by paying the full statement balance by the due date. Late payment fees are $29-$40; one-time waivers are available for customers with a strong payment history.",
    },
    "lost_stolen_cards": {
        "chunks": [
            """Reporting Lost or Stolen Cards
Report immediately via mobile app, online banking, or phone (24/7)
Card is immediately deactivated; new card ordered automatically
Expedited shipping: 2-3 business days ($15); standard: 5-7 business days (free)
Virtual card number available instantly via mobile app""",
            """Temporary Lock vs Permanent Deactivation
Temporary lock: card misplaced but might be found; blocks all transactions; can unlock immediately; automatic payments still process
Permanent deactivation: card truly lost or stolen; new card number issued; automatic payments need updating""",
            """Emergency Options
Emergency cash advance at any branch (subject to cash advance fees)
Temporary card available at select branches
International: call Chase international collect line (24/7); emergency card delivered in 3-5 days
Remember to update automatic payments with new card number after replacement""",
        ],
        "answer": "Lost or stolen cards should be reported immediately via the mobile app, online banking, or phone to prevent unauthorized charges. A temporary lock can be used when the card might be found; permanent deactivation is for confirmed lost/stolen cards. Replacement cards arrive in 5-7 business days (or 2-3 expedited). Virtual card numbers are available instantly.",
    },
}


def main():
    rows = []
    for topic_key, topic_data in TOPICS.items():
        topic_name = topic_key.replace("_", " ")
        for chunk in topic_data["chunks"]:
            rows.append({
                "question": topic_name,
                "retrieved_chunks": chunk,
                "answer": topic_data["answer"],
                "cited_chunks": json.dumps([chunk]),
            })

    _OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(_OUTPUT, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["question", "retrieved_chunks", "answer", "cited_chunks"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows across {len(TOPICS)} topics → {_OUTPUT}")


if __name__ == "__main__":
    main()
