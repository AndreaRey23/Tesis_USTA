import math

from poppy_torso_control.srv import *
from poppy_torso_control.msg import *

def plan2mov(message, fps):
    mov = {}
    mov['fps'] = fps
    mov['data'] = {}
    for plan in message.plans:
        mov['data'][plan.joint] = plan.trajectory
    return mov
