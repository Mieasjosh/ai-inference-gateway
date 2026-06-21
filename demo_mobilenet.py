#!/usr/bin/env python3
"""MobileNet-v2 图像分类 Demo — 验证 ONNX Runtime 真实推理

用法:
  python3 demo_mobilenet.py                     # 默认：随机噪声分类（Python onnxruntime 本地推理）
  python3 demo_mobilenet.py cat.jpg             # 指定图片文件
  python3 demo_mobilenet.py --server http://127.0.0.1:9993/infer  # 通过 C++ 网关推理
"""

import json
import sys
import os
import argparse
import time

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "mobilenetv2-7-clean.onnx")

# ImageNet 类别名（1000 类，前 10 摘要）
CLASS_NAMES = [
    "tench", "goldfish", "great white shark", "tiger shark", "hammerhead",
    "electric ray", "stingray", "cock", "hen", "ostrich",
    "Brambling", "goldfinch", "house finch", "junco", "indigo bunting",
    "robin", "bulbul", "jay", "magpie", "chickadee",
    "water ouzel", "kite", "bald eagle", "vulture", "great grey owl",
    "fire salamander", "smooth newt", "newt", "spotted salamander", "axolotl",
    "bullfrog", "tree frog", "tailed frog", "loggerhead", "leatherback turtle",
    "mud turtle", "terrapin", "box turtle", "banded gecko", "common iguana",
    "American chameleon", "whiptail", "agama", "frilled lizard", "alligator lizard",
    "Gila monster", "green lizard", "African chameleon", "Komodo dragon", "African crocodile",
    "American alligator", "triceratops", "thunder snake", "ringneck snake", "hognose snake",
    "green snake", "king snake", "garter snake", "water snake", "vine snake",
    "night snake", "boa constrictor", "rock python", "Indian cobra", "green mamba",
    "sea snake", "horned viper", "diamondback", "sidewinder", "trilobite",
    "harvestman", "scorpion", "black and gold garden spider", "barn spider", "garden spider",
    "black widow", "tarantula", "wolf spider", "tick", "centipede",
    "black grouse", "ptarmigan", "ruffed grouse", "prairie chicken", "peacock",
    "quail", "partridge", "African grey", "macaw", "sulphur-crested cockatoo",
    "lorikeet", "coucal", "bee eater", "hornbill", "hummingbird",
    "jacamar", "toucan", "drake", "red-breasted merganser", "goose",
    "black swan", "tusker", "echidna", "platypus", "wallaby",
    "koala", "wombat", "jellyfish", "sea anemone", "brain coral",
    "flatworm", "nematode", "conch", "snail", "slug",
    "sea slug", "chiton", "chambered nautilus", "Dungeness crab", "rock crab",
    "fiddler crab", "king crab", "American lobster", "spiny lobster", "crayfish",
    "hermit crab", "isopod", "white stork", "black stork", "spoonbill",
    "flamingo", "little blue heron", "American egret", "bittern", "crane",
    "limpkin", "European gallinule", "American coot", "bustard", "ruddy turnstone",
    "red-backed sandpiper", "redshank", "dowitcher", "oystercatcher", "pelican",
    "king penguin", "albatross", "grey whale", "killer whale", "dugong",
    "sea lion", "Chihuahua", "Japanese spaniel", "Maltese dog", "Pekinese",
    "Shih-Tzu", "Blenheim spaniel", "papillon", "toy terrier", "Rhodesian ridgeback",
    "Afghan hound", "basset", "beagle", "bloodhound", "bluetick",
    "black-and-tan coonhound", "Walker hound", "English foxhound", "redbone", "borzoi",
    "Irish wolfhound", "Italian greyhound", "whippet", "Ibizan hound", "Norwegian elkhound",
    "otterhound", "Saluki", "Scottish deerhound", "Weimaraner", "Staffordshire bullterrier",
    "American Staffordshire terrier", "Bedlington terrier", "Border terrier", "Kerry blue terrier", "Irish terrier",
    "Norfolk terrier", "Norwich terrier", "Yorkshire terrier", "wire-haired fox terrier", "Lakeland terrier",
    "Sealyham terrier", "Airedale", "cairn", "Australian terrier", "Dandie Dinmont",
    "Boston bull", "miniature schnauzer", "giant schnauzer", "standard schnauzer", "Scotch terrier",
    "Tibetan terrier", "silky terrier", "soft-coated wheaten terrier", "West Highland white terrier", "Lhasa",
    "flat-coated retriever", "curly-coated retriever", "golden retriever", "Labrador retriever", "Chesapeake Bay retriever",
    "German short-haired pointer", "vizsla", "English setter", "Irish setter", "Gordon setter",
    "Brittany spaniel", "clumber", "English springer", "Welsh springer spaniel", "cocker spaniel",
    "Sussex spaniel", "Irish water spaniel", "kuvasz", "schipperke", "groenendael",
    "malinois", "briard", "kelpie", "komondor", "Old English sheepdog",
    "Shetland sheepdog", "collie", "Border collie", "Bouvier des Flandres", "Rottweiler",
    "German shepherd", "Doberman", "miniature pinscher", "Greater Swiss Mountain dog", "Bernese mountain dog",
    "Appenzeller", "EntleBucher", "boxer", "bull mastiff", "Tibetan mastiff",
    "French bulldog", "Great Dane", "Saint Bernard", "Eskimo dog", "malamute",
    "Siberian husky", "dalmatian", "affenpinscher", "basenji", "pug",
    "Leonberg", "Newfoundland", "Great Pyrenees", "Samoyed", "Pomeranian",
    "chow", "keeshond", "Brabancon griffon", "Pembroke", "Cardigan",
    "toy poodle", "miniature poodle", "standard poodle", "Mexican hairless", "timber wolf",
    "white wolf", "red wolf", "coyote", "dingo", "dhole",
    "African hunting dog", "hyena", "red fox", "kit fox", "Arctic fox",
    "grey fox", "tabby", "tiger cat", "Persian cat", "Siamese cat",
    "Egyptian cat", "cougar", "lynx", "leopard", "snow leopard",
    "jaguar", "lion", "tiger", "cheetah", "brown bear",
    "American black bear", "ice bear", "sloth bear", "mongoose", "meerkat",
    "tiger beetle", "ladybug", "ground beetle", "long-horned beetle", "leaf beetle",
    "dung beetle", "rhinoceros beetle", "weevil", "fly", "bee",
    "ant", "grasshopper", "cricket", "walking stick", "cockroach",
    "mantis", "cicada", "leafhopper", "lacewing", "dragonfly",
    "damselfly", "admiral", "ringlet", "monarch", "cabbage butterfly",
    "sulphur butterfly", "lycaenid", "starfish", "sea urchin", "sea cucumber",
    "wood rabbit", "hare", "Angora", "hamster", "porcupine",
    "fox squirrel", "marmot", "beaver", "guinea pig", "sorrel",
    "zebra", "hog", "wild boar", "warthog", "hippopotamus",
    "ox", "water buffalo", "bison", "ram", "bighorn",
    "ibex", "hartebeest", "impala", "gazelle", "Arabian camel",
    "llama", "weasel", "mink", "polecat", "black-footed ferret",
    "otter", "skunk", "badger", "armadillo", "three-toed sloth",
    "orangutan", "gorilla", "chimpanzee", "gibbon", "siamang",
    "guenon", "patas", "baboon", "macaque", "langur",
    "colobus", "proboscis monkey", "marmoset", "capuchin", "howler monkey",
    "titi", "spider monkey", "squirrel monkey", "Madagascar cat", "indri",
    "Indian elephant", "African elephant", "lesser panda", "giant panda", "barracouta",
    "eel", "coho", "rock beauty", "anemone fish", "sturgeon",
    "gar", "lionfish", "puffer", "abacus", "abaya",
    "academic gown", "accordion", "acoustic guitar", "aircraft carrier", "airliner",
    "airship", "altar", "ambulance", "amphibian", "analog clock",
    "apiary", "apron", "ashcan", "assault rifle", "backpack",
    "bakery", "balance beam", "balloon", "ballpoint", "Band Aid",
    "banjo", "bannister", "barbell", "barber chair", "barbershop",
    "barn", "barometer", "barrel", "barrow", "baseball",
    "basketball", "bassinet", "bassoon", "bathing cap", "bath towel",
    "bathtub", "beach wagon", "beacon", "beaker", "bearskin",
    "beer bottle", "beer glass", "bell cote", "bib", "bicycle-built-for-two",
    "bikini", "binder", "binoculars", "birdhouse", "boathouse",
    "bobsled", "bolo tie", "bonnet", "bookcase", "bookshop",
    "bottlecap", "bow", "bow tie", "brass", "brassiere",
    "breakwater", "breastplate", "broom", "bucket", "buckle",
    "bulletproof vest", "bullet train", "butcher shop", "cab", "caldron",
    "candle", "cannon", "canoe", "can opener", "cardigan",
    "car mirror", "carousel", "carpenter's kit", "carton", "car wheel",
    "cash machine", "cassette", "cassette player", "castle", "catamaran",
    "CD player", "cello", "cellular telephone", "chain", "chainlink fence",
    "chain mail", "chain saw", "chest", "chiffonier", "chime",
    "china cabinet", "Christmas stocking", "church", "cinema", "cleaver",
    "cliff dwelling", "cloak", "clog", "cocktail shaker", "coffee mug",
    "coffeepot", "coil", "combination lock", "computer keyboard", "confectionery",
    "container ship", "convertible", "corkscrew", "cornet", "cowboy boot",
    "cowboy hat", "cradle", "crane (machine)", "crash helmet", "crate",
    "crib", "Crock Pot", "croquet ball", "crutch", "cuirass",
    "dam", "desk", "desktop computer", "dial telephone", "diaper",
    "digital clock", "digital watch", "dining table", "dishrag", "dishwasher",
    "disk brake", "dock", "dogsled", "dome", "doormat",
    "drilling platform", "drum", "drumstick", "dumbbell", "Dutch oven",
    "electric fan", "electric guitar", "electric locomotive", "entertainment center", "envelope",
    "espresso maker", "face powder", "feather boa", "file", "fireboat",
    "fire engine", "fire screen", "flagpole", "flute", "folding chair",
    "football helmet", "forklift", "fountain", "fountain pen", "four-poster",
    "freight car", "French horn", "frying pan", "fur coat", "garbage truck",
    "gasmask", "gas pump", "goblet", "go-kart", "golf ball",
    "golfcart", "gondola", "gong", "gown", "grand piano",
    "greenhouse", "grille", "grocery store", "guillotine", "hair slide",
    "hair spray", "half track", "hammer", "hamper", "hand blower",
    "hand-held computer", "handkerchief", "hard disc", "harmonica", "harp",
    "harvester", "hatchet", "holster", "home theater", "honeycomb",
    "hook", "hoopskirt", "horizontal bar", "horse cart", "hourglass",
    "iPod", "iron", "jack-o'-lantern", "jean", "jeep",
    "jersey", "jigsaw puzzle", "jinrikisha", "joystick", "kimono",
    "knee pad", "knot", "lab coat", "ladle", "lampshade",
    "laptop", "lawn mower", "lens cap", "letter opener", "library",
    "lifeboat", "lighter", "limousine", "liner", "lipstick",
    "Loafer", "lotion", "loudspeaker", "loupe", "lumbermill",
    "magnetic compass", "mailbag", "mailbox", "maillot (tights)", "maillot (tank suit)",
    "manhole cover", "maraca", "marimba", "mask", "matchstick",
    "maypole", "maze", "measuring cup", "medicine chest", "megalith",
    "microphone", "microwave", "military uniform", "milk can", "minibus",
    "miniskirt", "minivan", "missile", "mitten", "mixing bowl",
    "mobile home", "Model T", "modem", "monastery", "monitor",
    "moped", "mortar (vessel)", "mortarboard", "mosque", "mosquito net",
    "motor scooter", "mountain bike", "mountain tent", "mouse", "mousetrap",
    "moving van", "muzzle", "nail", "neck brace", "necklace",
    "nipple", "notebook", "obelisk", "oboe", "ocarina",
    "odometer", "oil filter", "organ", "oscilloscope", "overskirt",
    "oxcart", "oxygen mask", "packet", "paddle", "paddlewheel",
    "padlock", "paintbrush", "pajama", "palace", "panpipe",
    "paper towel", "parachute", "parallel bars", "park bench", "parking meter",
    "passenger car", "patio", "pay-phone", "pedestal", "pencil box",
    "pencil sharpener", "perfume", "Petri dish", "photocopier", "pick",
    "pickelhaube", "picket fence", "pickup", "pier", "piggy bank",
    "pill bottle", "pillow", "ping-pong ball", "pinwheel", "pirate",
    "pitcher", "plane", "planetarium", "plastic bag", "plate rack",
    "plow", "plunger", "Polaroid camera", "pole", "police van",
    "poncho", "pool table", "pop bottle", "pot", "potter's wheel",
    "power drill", "prayer rug", "printer", "prison", "projectile",
    "projector", "puck", "punching bag", "purse", "quill",
    "quilt", "racer", "racket", "radiator", "radio",
    "radio telescope", "rain barrel", "recreational vehicle", "reel", "reflex camera",
    "refrigerator", "remote control", "restaurant", "revolver", "rifle",
    "rocking chair", "rotisserie", "rubber eraser", "rugby ball", "rule",
    "running shoe", "safe", "safety pin", "saltshaker", "sandal",
    "sarong", "sax", "scabbard", "scale", "school bus",
    "schooner", "scoreboard", "screen", "screw", "screwdriver",
    "seat belt", "sewing machine", "shield", "shoe shop", "shoji",
    "shopping basket", "shopping cart", "shovel", "shower cap", "shower curtain",
    "ski", "ski mask", "sleeping bag", "slide rule", "sliding door",
    "slot", "snorkel", "snowmobile", "snowplow", "soap dispenser",
    "soccer ball", "sock", "solar dish", "sombrero", "soup bowl",
    "space bar", "space heater", "space shuttle", "spatula", "speedboat",
    "spider web", "spindle", "sports car", "spotlight", "stage",
    "steam locomotive", "steel arch bridge", "steel drum", "stethoscope", "stole",
    "stone wall", "stopwatch", "stove", "strainer", "streetcar",
    "stretcher", "studio couch", "stupa", "submarine", "suit",
    "sundial", "sunglass", "sunglasses", "sunscreen", "suspension bridge",
    "swab", "sweatshirt", "swimming trunks", "swing", "switch",
    "syringe", "table lamp", "tank", "tape player", "teapot",
    "teddy", "television", "tennis ball", "thatch", "theater curtain",
    "thimble", "thresher", "throne", "tile roof", "toaster",
    "tobacco shop", "toilet seat", "torch", "totem pole", "tow truck",
    "toyshop", "tractor", "trailer truck", "tray", "trench coat",
    "tricycle", "trimaran", "tripod", "triumphal arch", "trolleybus",
    "trombone", "tub", "turnstile", "typewriter keyboard", "umbrella",
    "unicycle", "upright", "vacuum", "vase", "vault",
    "velvet", "vending machine", "vestment", "viaduct", "violin",
    "volleyball", "waffle iron", "wall clock", "wallet", "wardrobe",
    "warplane", "washbasin", "washer", "water bottle", "water jug",
    "water tower", "whiskey jug", "whistle", "wig", "window screen",
    "window shade", "Windsor tie", "wine bottle", "wing", "wok",
    "wooden spoon", "wool", "worm fence", "wreck", "yawl",
    "yurt", "web site", "comic book", "crossword puzzle", "street sign",
    "traffic light", "book jacket", "menu", "plate", "guacamole",
    "consomme", "hot pot", "trifle", "ice cream", "ice lolly",
    "French loaf", "bagel", "pretzel", "cheeseburger", "hotdog",
    "mashed potato", "head cabbage", "broccoli", "cauliflower", "zucchini",
    "spaghetti squash", "acorn squash", "butternut squash", "cucumber", "artichoke",
    "bell pepper", "cardoon", "mushroom", "Granny Smith", "strawberry",
    "orange", "lemon", "fig", "pineapple", "banana",
    "jackfruit", "custard apple", "pomegranate", "hay", "carbonara",
    "chocolate sauce", "dough", "meat loaf", "pizza", "potpie",
    "burrito", "red wine", "espresso", "cup", "eggnog",
    "alp", "bubble", "cliff", "coral reef", "geyser",
    "lakeside", "promontory", "sandbar", "seashore", "valley",
    "volcano", "ballplayer", "groom", "scuba diver", "rapeseed",
    "daisy", "yellow lady's slipper", "corn", "acorn", "hip",
    "buckeye", "coral fungus", "agaric", "gyromitra", "stinkhorn",
    "earthstar", "hen-of-the-woods", "bolete", "ear", "toilet tissue",
]


def infer_local(input_data):
    """使用 Python onnxruntime 本地推理（无需 C++ 服务器）"""
    import numpy as np
    import onnxruntime as ort

    session = ort.InferenceSession(MODEL_PATH)
    arr = np.array(input_data, dtype=np.float32).reshape(1, 3, 224, 224)
    result = session.run(None, {"data": arr})
    return result[0][0].tolist()  # [1000] float


def infer_server(url, input_data):
    """通过 C++ 推理网关推理（适合小模型，如 test_model.onnx）"""
    import urllib.request
    payload = {"model": "mobilenet", "input": input_data}
    data = json.dumps(payload, separators=(',', ':')).encode()
    req = urllib.request.Request(url, data=data,
        headers={"Content-Type": "application/json"})
    resp = urllib.request.urlopen(req, timeout=60)
    return json.loads(resp.read()).get("output", [])


def preprocess_image(filepath):
    """加载图片并预处理为 MobileNet-v2 输入 [1, 3, 224, 224]，返回 flatten 列表"""
    try:
        from PIL import Image
    except ImportError:
        print("[ERROR] 需要 Pillow: pip install Pillow")
        return None
    import numpy as np

    img = Image.open(filepath).convert("RGB")
    img = img.resize((224, 224), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean) / std
    arr = arr.transpose(2, 0, 1)  # HWC → CHW
    return arr.flatten().tolist()


def random_input():
    import random
    return [random.uniform(-1, 1) for _ in range(3 * 224 * 224)]


def main():
    parser = argparse.ArgumentParser(description="MobileNet-v2 ONNX 推理 Demo")
    parser.add_argument("image", nargs="?", default=None,
                        help="图片文件路径（不传则使用随机噪声验证链路）")
    parser.add_argument("--server", default=None,
                        help="C++ 推理网关 URL（不传则使用 Python onnxruntime 本地推理）")
    parser.add_argument("--top", type=int, default=5,
                        help="显示 Top-N 分类结果 (default: 5)")
    args = parser.parse_args()

    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] 模型文件不存在: {MODEL_PATH}")
        print("  下载: python -c \"import urllib.request; ...\" ")
        sys.exit(1)

    print("=" * 60)
    print("MobileNet-v2 ONNX 推理 Demo")
    print("=" * 60)
    print(f"模型: {MODEL_PATH}")
    print(f"模式: {'C++ 网关' if args.server else 'Python onnxruntime 本地'}")

    # 准备输入
    if args.image:
        print(f"图片: {args.image}")
        input_data = preprocess_image(args.image)
        if input_data is None:
            sys.exit(1)
        print(f"  预处理: 224x224x3 → {len(input_data)} floats")
    else:
        print("输入: 随机噪声（验证链路畅通）")
        input_data = random_input()
        print(f"  生成 {len(input_data)} 个随机 float")

    print()

    # 执行推理
    try:
        start = time.perf_counter()
        if args.server:
            print(f"发送请求到 {args.server} ...")
            output = infer_server(args.server, input_data)
        else:
            print("本地 ONNX Runtime 推理中...")
            output = infer_local(input_data)
        elapsed = (time.perf_counter() - start) * 1000

        if not output or len(output) != 1000:
            print(f"[FAIL] 输出异常: 期望 1000 维, 实际 {len(output) if output else 0}")
            sys.exit(1)

        # Top-N 分类
        indexed = sorted(enumerate(output), key=lambda x: x[1], reverse=True)
        print(f"推理耗时: {elapsed:.0f}ms")
        print()
        print(f"Top-{args.top} 分类结果:")
        print("-" * 50)
        for rank, (idx, score) in enumerate(indexed[:args.top], 1):
            name = CLASS_NAMES[idx] if idx < len(CLASS_NAMES) else f"class_{idx}"
            bar = "█" * int(score * 50 / indexed[0][1]) if indexed[0][1] > 0 else ""
            print(f"  {rank:2d}. {name:<30s} {score:8.4f}  {bar}")
        print("-" * 50)
        print()
        print("ONNX Runtime 真实推理链路验证通过！")

    except Exception as e:
        print(f"[FAIL] {type(e).__name__}: {e}")
        if not args.server:
            print("  提示: pip install onnxruntime pillow")
        sys.exit(1)


if __name__ == "__main__":
    main()
