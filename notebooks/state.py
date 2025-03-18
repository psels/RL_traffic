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

def get_state_per_traffic_light(traffic_light):
    lane_ids = traci.trafficlight.getControlledLinks(traffic_light)
    print(lane_ids)
    values = []
    for value in lane_ids:
        for j in value:
            for k in j:
                values.append(k)
    lane_ids = values
    cleaned_lane_ids = [lane for lane in lane_ids if not lane.startswith(':')]
    return [traci.lane.getLastStepHaltingNumber(lane_id) for lane_id in cleaned_lane_ids] +\
        [traci.lane.getLastStepVehicleNumber(lane_id) for lane_id in cleaned_lane_ids]
