from setup.config import client

# Dataset examples
email_input_1 = {
    "author": "Alice Smith <alice.smith@company.com>",
    "to": "Robert Xu <Robert@company.com>",
    "subject": "Quick question about API documentation",
    "email_thread": """Hi Robert,

I was reviewing the API documentation for the new authentication service and noticed a few endpoints seem to be missing from the specs. Could you help clarify if this was intentional or if we should update the docs?

Specifically, I'm looking at:
- /auth/refresh
- /auth/validate

Thanks!
Alice""",
}

email_input_2 = {
    "author": "Marketing Team <marketing@company.com>",
    "to": "Robert Xu <Robert@company.com>",
    "subject": "New Company Newsletter Available",
    "email_thread": """Hello Robert,

The latest edition of our company newsletter is now available on the intranet. This month features articles on our Q2 results, upcoming team building activities, and employee spotlights.

Check it out when you have a chance!

Best regards,
Marketing Team""",
}

email_input_3 = {
    "author": "System Admin <sysadmin@company.com>",
    "to": "Robert Xu <Robert@company.com>",
    "subject": "Scheduled maintenance - database downtime",
    "email_thread": """Hi Robert,

This is a reminder that we'll be performing scheduled maintenance on the production database tonight from 2AM to 4AM EST. During this time, all database services will be unavailable.

Please plan your work accordingly and ensure no critical deployments are scheduled during this window.

Thanks,
System Admin Team""",
}

email_input_4 = {
    "author": "Project Manager <pm@client.com>",
    "to": "Robert Xu <Robert@company.com>",
    "subject": "Tax season let's schedule call",
    "email_thread": """Robert,

It's tax season again, and I wanted to schedule a call to discuss your tax planning strategies for this year. I have some suggestions that could potentially save you money.

Are you available sometime next week? Tuesday or Thursday afternoon would work best for me, for about 45 minutes.

Regards,
Project Manager""",
}

email_input_5 = {
    "author": "HR Department <hr@company.com>",
    "to": "Robert Xu <Robert@company.com>",
    "subject": "Reminder: Submit your expense reports",
    "email_thread": """Hello Robert,

This is a friendly reminder that all expense reports for the previous month need to be submitted by this Friday. Please make sure to include all receipts and proper documentation.

If you have any questions about the submission process, feel free to reach out to the HR team.

Best regards,
HR Department""",
}

email_input_6 = {
    "author": "Conference Organizer <events@techconf.com>",
    "to": "Robert Xu <Robert@company.com>",
    "subject": "Do you want to attend this conference?",
    "email_thread": """Hi Robert,

We're reaching out to invite you to TechConf 2025, happening May 15-17 in San Francisco. 

The conference features keynote speakers from major tech companies, workshops on AI and ML, and great networking opportunities. Early bird registration is available until April 30th.

Would you be interested in attending? We can also arrange for group discounts if other team members want to join.

Best regards,
Conference Organizers""",
}

email_input_7 = {
    "author": "Sarah Johnson <sarah.j@partner.com>",
    "to": "Robert Xu <Robert@company.com>",
    "subject": "Can you review these docs before submission?",
    "email_thread": """Robert,

I've attached the final version of our proposal for the Henderson project. Could you please review the technical specifications section (pages 15-20) before we submit it to the client on Friday?

Your expertise would really help ensure we've covered all the necessary details.

Thanks in advance,
Sarah""",
}

email_input_8 = {
    "author": "Community Pool <info@cityrecreation.org>",
    "to": "Robert Xu <Robert@company.com>",
    "subject": "Sign up daughter for swimming class",
    "email_thread": """Dear Robert,

Summer swimming registration is now open! Based on your daughter's participation last year, we wanted to let you know that intermediate level classes are available on Mondays and Wednesdays at 4PM or Tuesdays and Thursdays at 5PM.

Classes begin June 1st and run for 8 weeks. Space is limited, so early registration is recommended.

Please let us know if you'd like to reserve a spot.

Regards,
City Recreation Department""",
}

email_input_9 = {
    "author": "GitHub <notifications@github.com>",
    "to": "Robert Xu <Robert@company.com>",
    "subject": "PR #42: Comment from alex-dev",
    "email_thread": """Hey there!

alex-dev commented on your pull request #42 in langchain-ai/project:

> I've reviewed the changes and everything looks good. Just one small suggestion for the error handling in auth_controller.py. Maybe we should add a timeout parameter to prevent hanging requests?

View the comment: https://github.com/langchain-ai/project/pull/42#comment-12345

---
You're receiving this because you authored the thread.
Reply to this email directly, or view it on GitHub
""",
}

email_input_10 = {
    "author": "Team Lead <teamlead@company.com>",
    "to": "Robert Xu <Robert@company.com>",
    "subject": "Quarterly planning meeting",
    "email_thread": """Hi Robert,

It's time for our quarterly planning session. I'd like to schedule a 90-minute meeting next week to discuss our roadmap for Q3.

Could you let me know your availability for Monday or Wednesday? Ideally sometime between 10AM and 3PM.

Looking forward to your input on the new feature priorities.

Best,
Team Lead""",
}

email_input_11 = {
    "author": "AWS Monitoring <no-reply@aws.amazon.com>",
    "to": "Robert Xu <Robert@company.com>",
    "subject": "System admin alert: Instance CPU utilization exceeds threshold",
    "email_thread": """ALERT: High CPU Utilization

The following EC2 instance has exceeded the CPU utilization threshold of 90% for more than 15 minutes:

Instance ID: i-0b2d3e4f5a6b7c8d9
Region: us-west-2
Current utilization: 95.3%

This message is automatically generated. Please do not reply.
""",
}

email_input_12 = {
    "author": "Client Success <success@vendor.com>",
    "to": "Robert Xu <Robert@company.com>",
    "subject": "Your subscription will renew automatically",
    "email_thread": """Hello Robert,

This is a friendly reminder that your annual subscription to our Developer Pro plan will automatically renew on 04/15/2025.

Your payment method ending in **** 4567 will be charged $1,499.00.

If you would like to make any changes to your subscription, please visit your account settings or contact our support team before the renewal date.

Thank you for your continued business!

Client Success Team""",
}

email_input_13 = {
    "author": "Dr. Roberts <droberts@medical.org>",
    "to": "Robert Xu <Robert@company.com>",
    "subject": "Annual checkup reminder",
    "email_thread": """Hello Robert,

This is a reminder that it's time for your annual checkup. Our records show that your last visit was approximately one year ago.

Please call our office at (555) 123-4567 to schedule an appointment at your earliest convenience.

Best regards,
Dr. Roberts' Office""",
}

email_input_14 = {
    "author": "Social Media Platform <notifications@social.com>",
    "to": "Robert Xu <Robert@company.com>",
    "subject": "5 people liked your post",
    "email_thread": """Hi Robert,

5 people liked your recent post about "Machine Learning Techniques for NLP"

See who liked your post and continue the conversation!

[View activity]

To unsubscribe from these notifications, adjust your settings here.
""",
}

email_input_15 = {
    "author": "Project Team <project@company.com>",
    "to": "Robert Xu <Robert@company.com>",
    "subject": "Joint presentation next month",
    "email_thread": """Hi Robert,

The leadership team has asked us to prepare a joint presentation on our recent project successes for the all-hands meeting next month.

I've started putting together some slides and would appreciate your input on the technical architecture section. Could we schedule about 60 minutes sometime in the next week to collaborate on this?

I'm generally free on Tuesdays and Thursdays.

Thanks,
Project Team""",
}

email_input_16 = {
    "author": "Marketing Team <marketing@openai.com>",
    "to": "Robert Xu <Robert@company.com>",
    "subject": "Newsletter: New Model from OpenAI",
    "email_thread": """Hi Robert,

We're excited to announce that we've released a new model from OpenAI!

It's called "GPT-5" and it's a successor to GPT-4.

It's available now and you can find more information [here](https://openai.com/gpt-5).

Thanks,
Marketing Team""",
}

# Triage outputs: "ignore", "notify", "respond"
triage_output_1 = "respond"
triage_output_2 = "ignore"
triage_output_3 = "notify"
triage_output_4 = "respond"
triage_output_5 = "notify"
triage_output_6 = "respond"
triage_output_7 = "respond"
triage_output_8 = "respond"
triage_output_9 = "notify"
triage_output_10 = "respond"
triage_output_11 = "notify"
triage_output_12 = "notify"
triage_output_13 = "respond"
triage_output_14 = "ignore"
triage_output_15 = "respond"
triage_output_16 = "notify"

# Response criteria (when applicable)
response_criteria_1 = """
• Send email with write_email tool call to acknowledge the question and confirm it will be investigated  
"""

response_criteria_2 = """
• No response needed
• Ensure this is ignored  
"""

response_criteria_3 = """
• No response needed
• Ensure the user is notified  
"""

response_criteria_4 = """
• Check calendar availability for Tuesday or Thursday afternoon next week with check_calendar_availability tool call 
• Confirm availability for a 45-minute meeting
• Send calendar invite with schedule_meeting tool call 
• Send email with write_email tool call to acknowledge tax planning request and notifying that a meeting has been scheduled  
"""

response_criteria_5 = """
• No response needed
• Ensure the user is notified  
"""

response_criteria_6 = """
• Express interest in attending TechConf 2025
• Ask specific questions about AI/ML workshops
• Inquire about group discount details
• Send email with write_email tool call to express interest in attending TechConf 2025, ask specific questions about AI/ML workshops, and inquire about group discount details
"""

response_criteria_7 = """
• Explicitly agree to review the technical specifications
• Acknowledge Friday deadline
• Send email with write_email tool call to explicitly agree to review the technical specifications and acknowledge Friday deadline
"""

response_criteria_8 = """
• Send email with write_email tool call to express interest in registering daughter for swimming class
"""

response_criteria_9 = """
• No response needed
• Ensure the user is notified  
"""

response_criteria_10 = """
• Check calendar for 90-minute meeting availability for Monday or Wednesday with check_calendar_availability tool call 
• Send email acknowledging the request and providing availability with write_email tool call  
"""

response_criteria_11 = """
• No response needed
• Ensure the user is notified  
"""

response_criteria_12 = """
• No response needed
• Ensure the user is notified  
"""

response_criteria_13 = """
• Acknowledge annual checkup reminder
• Send email with write_email tool call to acknowledge annual checkup reminder
"""

response_criteria_14 = """
• No response needed
• Ensure this is ignored  
"""

response_criteria_15 = """
• Check calendar for 60-minute meeting availability for Tuesday or Thursday with check_calendar_availability tool call 
• Send calendar invite with schedule_meeting tool call 
• Send email agreeing to collaborate on the joint presentation and notifying that a meeting has been scheduled with write_email tool call  
"""

response_criteria_16 = """
• No response needed
• Ensure the user is notified  
"""


email_inputs = [
        email_input_1, email_input_2, email_input_3, email_input_4, email_input_5,
        email_input_6, email_input_7, email_input_8, email_input_9, email_input_10,
        email_input_11, email_input_12, email_input_13, email_input_14, email_input_15,
        email_input_16
    ]

email_names = [
    "email_input_1", "email_input_2", "email_input_3", "email_input_4", "email_input_5",
    "email_input_6", "email_input_7", "email_input_8", "email_input_9", "email_input_10",
    "email_input_11", "email_input_12", "email_input_13", "email_input_14", "email_input_15",
    "email_input_16"
]

response_criteria_list = [
    response_criteria_1, response_criteria_2, response_criteria_3, response_criteria_4, response_criteria_5,
    response_criteria_6, response_criteria_7, response_criteria_8, response_criteria_9, response_criteria_10,
    response_criteria_11, response_criteria_12, response_criteria_13, response_criteria_14, response_criteria_15,
    response_criteria_16
]

triage_outputs_list = [
    triage_output_1, triage_output_2, triage_output_3, triage_output_4, triage_output_5,
    triage_output_6, triage_output_7, triage_output_8, triage_output_9, triage_output_10,
    triage_output_11, triage_output_12, triage_output_13, triage_output_14, triage_output_15,
    triage_output_16
]

# Define expected tool calls for each email response based on content analysis
# Options: write_email, schedule_meeting, check_calendar_availability, done
expected_tool_calls = [
    ["write_email", "done"],                                                 # email_input_1: API documentation question
    [],                                                                      # email_input_2: Newsletter notification - ignore
    [],                                                                      # email_input_3: System maintenance notification - notification only
    ["check_calendar_availability", "schedule_meeting", "write_email", "done"], # email_input_4: Tax call scheduling
    [],                                                                      # email_input_5: Expense report reminder - notification only
    ["write_email", "done"],                                                 # email_input_6: Conference invitation - needs response
    ["write_email", "done"],                                                 # email_input_7: Document review request
    ["write_email", "done"],                                                 # email_input_8: Swimming class registration
    [],                                                                      # email_input_9: GitHub PR comment - notification only
    ["check_calendar_availability", "write_email", "done"], # email_input_10: Planning meeting
    [],                                                                      # email_input_11: AWS alert - notification only
    [],                                                                      # email_input_12: Subscription renewal - ignore
    ["write_email", "done"],                                                 # email_input_13: Doctor appointment reminder
    [],                                                                      # email_input_14: Social media notification - no action needed
    ["check_calendar_availability", "schedule_meeting", "write_email", "done"], # email_input_15: Joint presentation
    [],                                                                      # email_input_16: Newsletter - notification only
]

next_action_input_1 = {
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "schedule_meeting",
        "parameters": {
          "type": "object",
          "required": [
            "attendees",
            "subject",
            "duration_minutes",
            "preferred_day",
            "start_time"
          ],
          "properties": {
            "subject": {
              "type": "string"
            },
            "attendees": {
              "type": "array",
              "items": {
                "type": "string"
              }
            },
            "start_time": {
              "type": "integer"
            },
            "preferred_day": {
              "type": "string",
              "format": "date-time"
            },
            "duration_minutes": {
              "type": "integer"
            }
          }
        },
        "description": "Schedule a calendar meeting."
      }
    },
    {
      "type": "function",
      "function": {
        "name": "check_calendar_availability",
        "parameters": {
          "type": "object",
          "required": [
            "day"
          ],
          "properties": {
            "day": {
              "type": "string"
            }
          }
        },
        "description": "Check calendar availability for a given day."
      }
    },
    {
      "type": "function",
      "function": {
        "name": "write_email",
        "parameters": {
          "type": "object",
          "required": [
            "to",
            "subject",
            "content"
          ],
          "properties": {
            "to": {
              "type": "string"
            },
            "content": {
              "type": "string"
            },
            "subject": {
              "type": "string"
            }
          }
        },
        "description": "Write and send an email."
      }
    },
    {
      "type": "function",
      "function": {
        "name": "Done",
        "parameters": {
          "type": "object",
          "required": [
            "done"
          ],
          "properties": {
            "done": {
              "type": "boolean"
            }
          }
        },
        "description": "E-mail has been sent."
      }
    }
  ],
  "messages": [
    {
      "role": "system",
      "content": "\n< Role >\nYou are a top-notch executive assistant who cares about helping your executive perform as well as possible.\n</ Role >\n\n< Tools >\nYou have access to the following tools to help manage communications and schedule:\n\n1. write_email(to, subject, content) - Send emails to specified recipients\n2. schedule_meeting(attendees, subject, duration_minutes, preferred_day, start_time) - Schedule calendar meetings\n3. check_calendar_availability(day) - Check available time slots for a given day\n</ Tools >\n\n< Instructions >\nWhen handling emails, follow these steps:\n1. Carefully analyze the email content and purpose\n3. For responding to the email, draft a response email with the write_email tool\n4. For meeting requests, use the check_calendar_availability tool to find open time slots\n5. To schedule a meeting, use the schedule_meeting tool with a datetime object for the preferred_day parameter\n   - Today's date is 2025-09-03 - use this for scheduling meetings accurately\n6. If you scheduled a meeting, then draft a short response email using the write_email tool\n7. After using the write_email tool, the task is complete\n8. If you have sent the email, then use the Done tool to indicate that the task is complete\n</ Instructions >\n\n< Background >\nI'm Robert, a software engineer at LangChain.\n</ Background >\n\n< Response Preferences >\nUse professional and concise language. If the e-mail mentions a deadline, make sure to explicitly acknowledge and reference the deadline in your response.\n\nWhen responding to technical questions that require investigation:\n- Clearly state whether you will investigate or who you will ask\n- Provide an estimated timeline for when you'll have more information or complete the task\n\nWhen responding to event or conference invitations:\n- Always acknowledge any mentioned deadlines (particularly registration deadlines)\n- If workshops or specific topics are mentioned, ask for more specific details about them\n- If discounts (group or early bird) are mentioned, explicitly request information about them\n- Don't commit \n\nWhen responding to collaboration or project-related requests:\n- Acknowledge any existing work or materials mentioned (drafts, slides, documents, etc.)\n- Explicitly mention reviewing these materials before or during the meeting\n- When scheduling meetings, clearly state the specific day, date, and time proposed.\n\nWhen responding to meeting scheduling requests:\n- If the recipient is asking for a meeting commitment, verify availability for all time slots mentioned in the original email and then commit to one of the proposed times based on your availability by scheduling the meeting. Or, say you can't make it at the time proposed.\n- If availability is asked for, then check your calendar for availability and send an email proposing multiple time options when available. Do NOT schedule meetings\n- Mention the meeting duration in your response to confirm you've noted it correctly.\n- Reference the meeting's purpose in your response.\n</ Response Preferences >\n\n< Calendar Preferences >\n30 minute meetings are preferred, but 15 minute meetings are also acceptable.\nTimes later in the day are preferable.\n</ Calendar Preferences >\n"
    },
    {
      "role": "user",
      "content": "Respond to the email: \n\n**Subject**: Annual checkup reminder\n**From**: Dr. Roberts <droberts@medical.org>\n**To**: Robert Xu <Robert@company.com>\n\nHello Robert,\n\nThis is a reminder that it's time for your annual checkup. Our records show that your last visit was approximately one year ago.\n\nPlease call our office at (555) 123-4567 to schedule an appointment at your earliest convenience.\n\nBest regards,\nDr. Roberts' Office\n\n---\n"
    }
  ]
}

next_action_input_2 = {
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "schedule_meeting",
        "parameters": {
          "type": "object",
          "required": [
            "attendees",
            "subject",
            "duration_minutes",
            "preferred_day",
            "start_time"
          ],
          "properties": {
            "subject": {
              "type": "string"
            },
            "attendees": {
              "type": "array",
              "items": {
                "type": "string"
              }
            },
            "start_time": {
              "type": "integer"
            },
            "preferred_day": {
              "type": "string",
              "format": "date-time"
            },
            "duration_minutes": {
              "type": "integer"
            }
          }
        },
        "description": "Schedule a calendar meeting."
      }
    },
    {
      "type": "function",
      "function": {
        "name": "check_calendar_availability",
        "parameters": {
          "type": "object",
          "required": [
            "day"
          ],
          "properties": {
            "day": {
              "type": "string"
            }
          }
        },
        "description": "Check calendar availability for a given day."
      }
    },
    {
      "type": "function",
      "function": {
        "name": "write_email",
        "parameters": {
          "type": "object",
          "required": [
            "to",
            "subject",
            "content"
          ],
          "properties": {
            "to": {
              "type": "string"
            },
            "content": {
              "type": "string"
            },
            "subject": {
              "type": "string"
            }
          }
        },
        "description": "Write and send an email."
      }
    },
    {
      "type": "function",
      "function": {
        "name": "Done",
        "parameters": {
          "type": "object",
          "required": [
            "done"
          ],
          "properties": {
            "done": {
              "type": "boolean"
            }
          }
        },
        "description": "E-mail has been sent."
      }
    }
  ],
  "messages": [
    {
      "role": "system",
      "content": "\n< Role >\nYou are a top-notch executive assistant who cares about helping your executive perform as well as possible.\n</ Role >\n\n< Tools >\nYou have access to the following tools to help manage communications and schedule:\n\n1. write_email(to, subject, content) - Send emails to specified recipients\n2. schedule_meeting(attendees, subject, duration_minutes, preferred_day, start_time) - Schedule calendar meetings\n3. check_calendar_availability(day) - Check available time slots for a given day\n</ Tools >\n\n< Instructions >\nWhen handling emails, follow these steps:\n1. Carefully analyze the email content and purpose\n3. For responding to the email, draft a response email with the write_email tool\n4. For meeting requests, use the check_calendar_availability tool to find open time slots\n5. To schedule a meeting, use the schedule_meeting tool with a datetime object for the preferred_day parameter\n   - Today's date is 2025-09-03 - use this for scheduling meetings accurately\n6. If you scheduled a meeting, then draft a short response email using the write_email tool\n7. After using the write_email tool, the task is complete\n8. If you have sent the email, then use the Done tool to indicate that the task is complete\n</ Instructions >\n\n< Background >\nI'm Robert, a software engineer at LangChain.\n</ Background >\n\n< Response Preferences >\nUse professional and concise language. If the e-mail mentions a deadline, make sure to explicitly acknowledge and reference the deadline in your response.\n\nWhen responding to technical questions that require investigation:\n- Clearly state whether you will investigate or who you will ask\n- Provide an estimated timeline for when you'll have more information or complete the task\n\nWhen responding to event or conference invitations:\n- Always acknowledge any mentioned deadlines (particularly registration deadlines)\n- If workshops or specific topics are mentioned, ask for more specific details about them\n- If discounts (group or early bird) are mentioned, explicitly request information about them\n- Don't commit \n\nWhen responding to collaboration or project-related requests:\n- Acknowledge any existing work or materials mentioned (drafts, slides, documents, etc.)\n- Explicitly mention reviewing these materials before or during the meeting\n- When scheduling meetings, clearly state the specific day, date, and time proposed.\n\nWhen responding to meeting scheduling requests:\n- If the recipient is asking for a meeting commitment, verify availability for all time slots mentioned in the original email and then commit to one of the proposed times based on your availability by scheduling the meeting. Or, say you can't make it at the time proposed.\n- If availability is asked for, then check your calendar for availability and send an email proposing multiple time options when available. Do NOT schedule meetings\n- Mention the meeting duration in your response to confirm you've noted it correctly.\n- Reference the meeting's purpose in your response.\n</ Response Preferences >\n\n< Calendar Preferences >\n30 minute meetings are preferred, but 15 minute meetings are also acceptable.\nTimes later in the day are preferable.\n</ Calendar Preferences >\n"
    },
    {
      "role": "user",
      "content": "Respond to the email: \n\n**Subject**: Joint presentation next month\n**From**: Project Team <project@company.com>\n**To**: Robert Xu <Robert@company.com>\n\nHi Robert,\n\nThe leadership team has asked us to prepare a joint presentation on our recent project successes for the all-hands meeting next month.\n\nI've started putting together some slides and would appreciate your input on the technical architecture section. Could we schedule about 60 minutes sometime in the next week to collaborate on this?\n\nI'm generally free on Tuesdays and Thursdays.\n\nThanks,\nProject Team\n\n---\n"
    },
    {
      "role": "assistant",
      "content": "",
      "tool_calls": [
        {
          "id": "call_ZqbN4kI62jHx1OBesZWtEwTG",
          "type": "function",
          "function": {
            "name": "check_calendar_availability",
            "arguments": "{\"day\": \"2025-09-09\"}"
          }
        }
      ]
    },
    {
      "role": "tool",
      "content": "Available times on 2025-09-09: 9:00 AM, 2:00 PM, 4:00 PM",
      "tool_call_id": "call_ZqbN4kI62jHx1OBesZWtEwTG"
    },
    {
      "role": "assistant",
      "content": "",
      "tool_calls": [
        {
          "id": "call_2MgCSNSnm3ijNc3omz9jfXun",
          "type": "function",
          "function": {
            "name": "check_calendar_availability",
            "arguments": "{\"day\": \"2025-09-11\"}"
          }
        }
      ]
    },
    {
      "role": "tool",
      "content": "Available times on 2025-09-11: 9:00 AM, 2:00 PM, 4:00 PM",
      "tool_call_id": "call_2MgCSNSnm3ijNc3omz9jfXun"
    }
  ]
}

next_action_output_1 = {
  "message": {
    "role": "assistant",
    "content": "",
    "tool_calls": [
      {
        "id": "call_f5Vw3fsrJp00zeBlJuosmjGa",
        "type": "function",
        "function": {
          "name": "write_email",
          "arguments": "{\"to\": \"droberts@medical.org\", \"subject\": \"Re: Annual checkup reminder\", \"content\": \"Hello Dr. Roberts' Office,\\n\\nThank you for the reminder regarding my annual checkup. I appreciate it. I will call your office soon to schedule an appointment at a convenient time.\\n\\nBest regards,\\nRobert Xu\"}"
        }
      }
    ]
  }
}

next_action_output_2 = {
  "message": {
    "role": "assistant",
    "content": "",
    "tool_calls": [
      {
        "id": "call_ja11giw2j4eH3xfYoq6OZMvy",
        "type": "function",
        "function": {
          "name": "schedule_meeting",
          "arguments": "{\"attendees\": [\"project@company.com\"], \"subject\": \"Collaboration on Technical Architecture Section for Joint Presentation\", \"duration_minutes\": 60, \"preferred_day\": \"2025-09-11\", \"start_time\": 14}"
        }
      }
    ]
  }
}
next_action_input_list = [next_action_input_1, next_action_input_2]
next_action_output_list = [next_action_output_1, next_action_output_2]

def load_triage_datasets():
    dataset_name = "Email Agent: Triage"
    if not client.has_dataset(dataset_name=dataset_name):
        dataset = client.create_dataset(dataset_name=dataset_name)
        client.create_examples(
            inputs=[{"email_input": email} for email in email_inputs],
            outputs=[{"classification": classification} for classification in triage_outputs_list],
            dataset_id=dataset.id,
        )
    return dataset_name

def load_response_datasets():
    dataset_name = "Email Agent: Final Response"
    if not client.has_dataset(dataset_name=dataset_name):
        dataset = client.create_dataset(dataset_name=dataset_name)
        client.create_examples(
            inputs=[{"email_input": email} for email in email_inputs],
            outputs=[{"response_criteria": response} for response in response_criteria_list],
            dataset_id=dataset.id,
        )
    return dataset_name

def load_trajectory_datasets():
    dataset_name = "Email Agent: Trajectory"
    if not client.has_dataset(dataset_name=dataset_name):
        dataset = client.create_dataset(dataset_name=dataset_name)
        client.create_examples(
            inputs=[{"email_input": email} for email in email_inputs],
            outputs=[{"trajectory": trajectory} for trajectory in expected_tool_calls],
            dataset_id=dataset.id,
        )
    return dataset_name

def load_next_action_datasets():
    dataset_name = "Email Agent: Next Action"
    if not client.has_dataset(dataset_name=dataset_name):
        dataset = client.create_dataset(dataset_name=dataset_name)
        client.create_examples(
            inputs=next_action_input_list,
            outputs=next_action_output_list,
            dataset_id=dataset.id,
        )
    return dataset_name

def load_datasets():
    print("Loading Datasets...")
    print("   - Triage Datasets...")
    load_triage_datasets()

    print("   - Response Datasets...")
    load_response_datasets()
    
    print("   - Trajectory Datasets...")
    load_trajectory_datasets()
    
    print("   - Next Action Datasets...")
    load_next_action_datasets()

if __name__ == "__main__":
    load_datasets()