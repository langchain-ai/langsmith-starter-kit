import argparse

from src.email_agent.use_case import EmailAgentUseCase
from src.finance_qa.use_case import FinanceQAUseCase

USE_CASES = {
    "email-agent": EmailAgentUseCase,
    "finance-qa": FinanceQAUseCase,
}


def main():
    parser = argparse.ArgumentParser(description="LangSmith Starter Kit")
    parser.add_argument(
        "--use-case",
        choices=list(USE_CASES.keys()),
        default="email-agent",
        help="Which use case to run (default: email-agent)",
    )
    parser.add_argument(
        "--admin",
        action="store_true",
        help="Run with admin privileges (setup workspace secrets).",
    )
    parser.add_argument(
        "--num-traces",
        type=int,
        default=None,
        help="Number of traces to generate (default: all for email-agent, 20 for chatbot).",
    )
    parser.add_argument(
        "--teardown",
        action="store_true",
        help="Delete all LangSmith resources for the use case and exit.",
    )
    args = parser.parse_args()

    use_case = USE_CASES[args.use_case]()
    if args.teardown:
        use_case.teardown()
    else:
        use_case.run(admin=args.admin, num_traces=args.num_traces)


if __name__ == "__main__":
    main()
