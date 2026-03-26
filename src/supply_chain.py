import numpy as np

def calculate_inventory(demand_series, lead_time=2, z=1.65):
    import numpy as np

    avg_demand = float(np.nan_to_num(demand_series.mean()))
    std_demand = float(np.nan_to_num(demand_series.std()))

    reorder_point = float(np.nan_to_num(avg_demand * lead_time))
    safety_stock = float(np.nan_to_num(z * std_demand * np.sqrt(lead_time)))
    inventory_level = float(np.nan_to_num(reorder_point + safety_stock))

    return {
        "avg_demand": avg_demand,
        "std_demand": std_demand,
        "reorder_point": reorder_point,
        "safety_stock": safety_stock,
        "inventory_level": inventory_level
    }

def simple_ahp(scores, weights):
    scores = np.array(scores)
    weights = np.array(weights)

    normalized = scores / scores.max(axis=0)
    final_scores = normalized.dot(weights)

    return final_scores

def demand_spike_analysis(demand_series, spike_factor=1.3):
    spike_demand = demand_series * spike_factor

    return {
        "original_mean": demand_series.mean(),
        "spike_mean": spike_demand.mean()
    }