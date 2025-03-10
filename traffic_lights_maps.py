import traci

def get_traffic_light_phases(traffic_light_id):
    programs = traci.trafficlight.getAllProgramLogics(traffic_light_id)
    if programs:
        return list(range(len(programs[0].phases)))  
    else:
        return [0] 

def make_map(trafic_light_ids):
    actions_map = {}
    for idx, traffic_light_id in enumerate(trafic_light_ids):
        actions_map[idx] = get_traffic_light_phases(traffic_light_id)
    return actions_map