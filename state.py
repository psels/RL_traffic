import traci

# def get_state(detector_ids):
#     state = []
#     for detector_id in detector_ids:
#         detector_state = traci.inductionloop.getLastStepOccupancy(detector_id) # on pourrait ajouter d'autres éléments
#         state.append(detector_state)
#     return state

def queue(lane_ids):
    state = []
    for lane_id in lane_ids:
        lane_state = traci.lane.getLastStepHaltingNumber(lane_id)
        state.append(lane_state)
    return state

def get_state(lane_ids):
    return [traci.lane.getLastStepHaltingNumber(lane_id) for i,lane_id in enumerate(lane_ids) if i>=12]+\
        [traci.lane.getLastStepVehicleNumber(lane_id) for i,lane_id in enumerate(lane_ids) if i>=12]
