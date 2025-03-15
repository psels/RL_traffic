import traci


def queue(lane_ids):
    state = []
    for lane_id in lane_ids:
        lane_state = traci.lane.getLastStepHaltingNumber(lane_id)
        state.append(lane_state)
    return state

def get_state(lane_ids):
    cleaned_lane_ids = [lane for lane in lane_ids if not lane.startswith(':')]

    return [traci.lane.getLastStepHaltingNumber(lane_id) for lane_id in cleaned_lane_ids] +\
        [traci.lane.getLastStepVehicleNumber(lane_id) for lane_id in cleaned_lane_ids]
