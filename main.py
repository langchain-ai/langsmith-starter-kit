import argparse

from src.email_agent.use_case import EmailAgentUseCase
from src.chatbot.use_case import ChatbotUseCase

USE_CASES = {
    "email-agent": EmailAgentUseCase,
    "chatbot": ChatbotUseCase,
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
        "--api-only",
        action="store_true",
        help="Use LangSmith REST API instead of SDK where supported.",
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
    args = parser.parse_args()

    use_case = USE_CASES[args.use_case](use_api=args.api_only)
    use_case.run(admin=args.admin, num_traces=args.num_traces)


if __name__ == "__main__":
    main()
