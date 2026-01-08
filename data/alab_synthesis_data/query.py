import pandas as pd
from pymongo import MongoClient
import re

# Connect to MongoDB (adjust URI/port as needed)

def connect_to_sauron():
    """MongoDB connection details -- Sauron"""
    db= MongoClient(host="mongodb07.nersc.gov", 
                        username="alab_completed_ro",
                        password="CEDERALAB_RO", 
                        authSource="alab_completed")["alab_completed"]
    collection= db['samples']
    return collection
# Main logic
def main():
    # Ask user for project name (e.g., ARR)
    project = input("Enter project name (e.g., ARR): ").strip() #.upper()
    
    # Create regex query to match sample names like ARR_12345
    regex = re.compile(f"^{project}_")  # Match any name starting with ARR_, etc.
    query = {"name": regex}
    # regex = f"^{project}_\\d{{5}}"  # Adjust pattern if needed
    # query = {"name": {"$regex": regex}}

    # Connect to sauron DB
    collection = connect_to_sauron()

    # Fetch matching documents
    documents = list(collection.find(query))

    rows = []
    for doc in documents:
        # name = doc.get("name", "")
        name = doc.get("name", "").replace("_", "-")  # replace underscore with dash
        metadata = doc.get("metadata", {})

        # Extract target
        target = metadata.get("target")

        # Precursors from powder dosing (if available)
        powder_dosing = metadata.get("powderdosing_results", {})
        if isinstance(powder_dosing, dict):
            powders = powder_dosing.get("Powders", [])
        else:
            powders = []
        precursors = [p.get("PowderName", "") for p in powders]
        # precursors_str = ", ".join(precursors)
        precursors_str = str(precursors)

        # Temperature & dwell time
        heating = metadata.get("heating_results", {})
        temp = heating.get("heating_temperature")
        dwell_s = heating.get("heating_time")
        dwell_hr = round(dwell_s / 3600, 2) if isinstance(dwell_s, (int, float)) else None

        # Furnace info (if available)
        furnace = heating.get("furnace_name", "BF")

        rows.append({
            "Name": name,
            "Target": target,
            "Precursors": precursors_str,
            "Temperature (C)": temp,
            "Dwell Duration (h)": dwell_hr,
            "Furnace": furnace
        })

    # Create DataFrame
    df = pd.DataFrame(rows)
    
    # Save CSV
    filename = f"synthesis_{project}.csv"
    df.to_csv(filename, index=False)
    print(f"Saved {len(df)} entries to {filename}")

if __name__ == "__main__":
    main()