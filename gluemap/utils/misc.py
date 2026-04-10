def get_tracks_dict_indexes(tracks_dict):
    if isinstance(tracks_dict["indexes"], dict):
        indexes = list(tracks_dict["indexes"].keys())
    elif isinstance(tracks_dict["indexes"], list):
        indexes = list(range(len(tracks_dict["indexes"])))

    return indexes
