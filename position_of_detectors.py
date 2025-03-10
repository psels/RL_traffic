import traci
import folium


sumoBinary = r"C:/Program Files/rl_project/Eclipse/Sumo/bin/sumo.exe" 
sumoConfig = r"C:/Program Files/rl_project/2025-02-12-16-24-37/osm.sumocfg"

sumoCmd = [sumoBinary, "-c", sumoConfig, "--start"]
traci.start(sumoCmd)

detector_data = {}

while traci.simulation.getMinExpectedNumber() > 0:
    traci.simulationStep() 

    for detector_id in traci.inductionloop.getIDList():
        lane_id = traci.inductionloop.getLaneID(detector_id)
        pos = traci.inductionloop.getPosition(detector_id)

        detector_data[detector_id] = (lane_id, pos)

geo_positions = {}

for detector_id, (lane_id, pos) in detector_data.items():
    try:
        lane_shape = traci.lane.getShape(lane_id)

        if not lane_shape:
            print(f"issues with {lane_id}")
            continue

        x, y = lane_shape[0] # c'est une approximation
        lon, lat = traci.simulation.convertGeo(x, y)

        geo_positions[detector_id] = (lat, lon)
    except Exception as e:
        print(f"Error with detector {detector_id}: {e}")

traci.close()

# carte Folium
if geo_positions:
    first_lat, first_lon = list(geo_positions.values())[0]
    map_center = (first_lat, first_lon)
else:
    map_center = (0, 0)

m = folium.Map(location=map_center, zoom_start=14)

for detector_id, (lat, lon) in geo_positions.items():
    folium.Marker(
        location=(lat, lon),
        popup=f"Detector {detector_id}",
        icon=folium.Icon(color="blue", icon="info-sign")
    ).add_to(m)

m.save("detector_map.html")