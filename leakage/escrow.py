#ESCROW GATING
#blocks forbidden-link messages until escrow is funded.


ESCROW_STATUS = {}  # task_id -> True/False

def is_escrow_funded(task_id: str):
    return ESCROW_STATUS.get(task_id, False)
