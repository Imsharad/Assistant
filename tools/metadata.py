import logging
from utils.time import timeit

general_logger = logging.getLogger("general")


@timeit()
def deploy_validation_rule(sf_conn, validation_rule):
    result = sf_conn.toolingexecute(
        method="POST", action="sobjects/ValidationRule", json=validation_rule
    )

    if result.get("success"):
        general_logger.info(f"Validation rule created successfully. ID: {result['id']}")
    else:
        general_logger.info(f"Failed to create validation rule. Error: {result}")

    return result
