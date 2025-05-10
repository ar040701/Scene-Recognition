import torch

# Load the checkpoint
checkpoint_path = "vit_base.pth"  # Replace with your actual file path
state_dict = torch.load(checkpoint_path, map_location="cpu")

# Check the classifier layer
if 'classifier.weight' in state_dict:
    num_classes = state_dict['classifier.weight'].shape[0]
    print(f"Number of classes: {num_classes}")

class_labels = [
    "airport_terminal", "amphitheatre","amusement_park", "art_gallery",
    "bakery_shop","bar","bookstore", "botanical_garden","bridge",
    "bus interior","butchers shop","campsite","classroom","coffee_shop",
    "construction_site","courtyard","driveway","fire_station","fountain",
    "gas_station","harbour","highway","kindergarten_classroom","lobby",
    "market_outdoor","museum","office","parking_lot","phone_booth",
    "playground","railroad_track","restaurant","river","shed","staircase",
    "supermarket","swomming_pool_outdoor","track","valley","yard"
]
print(len(class_labels) == num_classes)  # Should print True if the number of classes matches the checkpoint