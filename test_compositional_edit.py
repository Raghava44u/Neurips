import os
import torch
import sys
import json

from easyeditor.trainer.MultimodalTrainer import MultimodalTrainer
from easyeditor.dataset.coco_caption import CompositionalCaptionDataset, CompositionalDataset_RAG_70, CompositionalDataset_RAG_50, CompositionalDataset
from easyeditor.models.ft.ft_multimodal_hparams import FTMultimodalHparams
from easyeditor.models.lora.lora_multimodal_hparams import LORAMultimodalHparams
from easyeditor.models.ours.ours_hparams import OURSMultimodalHparams
from easyeditor.models.wise.wise_multimodal_hparams import WISEMultimodalHyperParams
from easyeditor.trainer.training_hparams.serac_multimodal_training_hparams import SERACMultimodalTrainingHparams

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
CUSTOM_EVAL_DATASET = os.environ.get(
    'MEMEIC_EVAL_DATASET',
    os.path.abspath(os.path.join(REPO_ROOT, '..', 'CCKEB_eval.json')),
)
CUSTOM_IMAGE_ROOT = os.environ.get(
    'MEMEIC_IMAGE_ROOT',
    os.path.join(REPO_ROOT, 'CCKEB_images', 'mmkb_images'),
)
DEFAULT_HOP = 1
RESULTS_ROOT = os.environ.get('MEMEIC_RESULTS_DIR', os.path.join(REPO_ROOT, 'results'))
DEFAULT_TEST_NUM = int(os.environ.get('MEMEIC_TEST_NUM', '500'))
DEFAULT_GAP_SCHEDULE = [0, 10, 20, 50, 100]
ENABLE_ARTIFACTS = os.environ.get('MEMEIC_ENABLE_ARTIFACTS', '1').strip().lower() not in {'0', 'false', 'no'}


def resolve_gap_schedule():
    raw_schedule = os.environ.get('MEMEIC_EVAL_GAPS', '').strip()
    if not raw_schedule:
        return DEFAULT_GAP_SCHEDULE

    gap_values = []
    for part in raw_schedule.split(','):
        part = part.strip()
        if not part:
            continue
        gap_values.append(int(part))

    return gap_values or DEFAULT_GAP_SCHEDULE


def resolve_cuda_gpus(preferred=None, required=2):
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for adversarial evaluation, but no GPU is available.')

    override = os.environ.get('MEMEIC_EVAL_GPUS')
    if override:
        preferred = [int(value.strip()) for value in override.split(',') if value.strip()]

    device_count = torch.cuda.device_count()
    if preferred is None:
        preferred = list(range(min(device_count, required)))

    valid = [gpu for gpu in preferred if 0 <= gpu < device_count]
    if not valid:
        valid = [0]

    while len(valid) < required:
        valid.append(valid[-1])
    return valid[:required]


def prepare_eval_hparams(hparams, preferred_gpus=None, required_gpus=1):
    gpus = resolve_cuda_gpus(preferred_gpus, required=required_gpus)
    hparams.device = gpus[0]
    hparams.results_dir = RESULTS_ROOT
    torch.cuda.set_device(gpus[0])
    return gpus


def make_artifact_options(model_name):
    return {
        'enabled': ENABLE_ARTIFACTS,
        'repo_root': REPO_ROOT,
        'results_root': RESULTS_ROOT,
        'model_name': model_name,
    }


def _normalize_rel_path(path_value):
    normalized = str(path_value).replace('\\', '/')
    if os.path.isabs(normalized):
        normalized = os.path.relpath(normalized, REPO_ROOT).replace('\\', '/')
    return normalized


def normalize_adversarial_dataset(dataset_path):
    with open(dataset_path, 'r', encoding='utf-8') as handle:
        rows = json.load(handle)

    changed = False
    required = {
        'src', 'rephrase', 'pred', 'alt', 'image', 'image_rephrase', 'loc', 'loc_ans',
        'm_loc', 'm_loc_q', 'm_loc_a', 'src_q', 'rephrase_q', 'm_loc_q_q', 'port_new', 'textual_edit'
    }

    for idx, row in enumerate(rows):
        missing = required.difference(row.keys())
        if missing:
            raise KeyError(f"Row {idx} is missing required keys: {sorted(missing)}")

        for key in ('image', 'image_rephrase', 'm_loc'):
            normalized = _normalize_rel_path(row[key])
            if normalized != row[key]:
                row[key] = normalized
                changed = True

        textual_edit = row['textual_edit']
        if isinstance(textual_edit.get('pred'), str):
            textual_edit['pred'] = [textual_edit['pred']]
            changed = True
        if isinstance(textual_edit.get('alt'), str):
            textual_edit['alt'] = [textual_edit['alt']]
            changed = True

        for field in ('image', 'image_rephrase', 'm_loc'):
            resolved = os.path.join(CUSTOM_IMAGE_ROOT, row[field])
            if not os.path.exists(resolved):
                raise FileNotFoundError(f"Missing image for row {idx}: {resolved}")

    if changed:
        with open(dataset_path, 'w', encoding='utf-8') as handle:
            json.dump(rows, handle, indent=2, ensure_ascii=False)

    return rows


def build_eval_dataset(dataset_cls, hparams, dataset_path, hop, debug=True):
    rows = normalize_adversarial_dataset(dataset_path)
    hparams.coco_image = CUSTOM_IMAGE_ROOT
    hparams.rephrase_image = CUSTOM_IMAGE_ROOT

    if debug and rows:
        first = rows[0]
        resolved_image = os.path.join(CUSTOM_IMAGE_ROOT, first['image'])
        print(f"[debug] Loaded benchmark: {dataset_path}")
        print(f"[debug] Total samples: {len(rows)}")
        print(f"[debug] Artifacts enabled: {ENABLE_ARTIFACTS}")
        print(f"[debug] First sample: {first}")
        print(f"[debug] Resolved image path: {resolved_image}")
        print(f"[debug] Image exists: {os.path.exists(resolved_image)}")

    dataset = dataset_cls(dataset_path, config=hparams, hop=hop)
    if debug and len(dataset) > 0:
        print(f"[debug] Dataset loaded samples: {len(dataset)}")
    return dataset

'''
Baselines
'''

####################### FT ##########################

def test_LLaVA_FT_comp():
    '''
    FT baseline for LLaVA compositional editing
    Uses CompositionalDataset (no Query Decomposition) + adversal2k.json
    '''
    hparams = FTMultimodalHparams.from_hparams('hparams/FT/llava_compositional_edit.yaml')
    prepare_eval_hparams(hparams, preferred_gpus=[0], required_gpus=1)
    eval_ds = build_eval_dataset(CompositionalDataset, hparams, eval_comp_final_json_path, hop)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=eval_ds,
        val_set=eval_ds
    )
    trainer.test_sequencial_compositional_ft(log=True, gap_num=gap_num, test_num=DEFAULT_TEST_NUM, artifact_options=make_artifact_options('FT-LLaVA'))

def test_MiniGPT4_FT_comp():
    '''
    FT baseline for MiniGPT4 compositional editing
    Uses CompositionalDataset (no Query Decomposition) + adversal2k.json
    '''
    hparams = FTMultimodalHparams.from_hparams('hparams/FT/minigpt4_compositional_edit.yaml')
    prepare_eval_hparams(hparams, preferred_gpus=[1], required_gpus=1)
    eval_ds = build_eval_dataset(CompositionalDataset, hparams, eval_comp_final_json_path, hop)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=eval_ds,
        val_set=eval_ds
    )
    trainer.test_sequencial_compositional_ft(log=True, gap_num=gap_num, test_num=DEFAULT_TEST_NUM, artifact_options=make_artifact_options('FT-MiniGPT4'))

####################### LoRA ##########################

def test_LLaVA_one_lora_comp():
    '''
    LoRA baseline for LLaVA compositional editing (single LoRA for both visual and textual)
    Uses CompositionalDataset (no Query Decomposition) + adversal2k.json
    '''
    hparams = LORAMultimodalHparams.from_hparams('hparams/LORA/llava_compositional_one_lora.yaml')
    prepare_eval_hparams(hparams, preferred_gpus=[0], required_gpus=1)
    eval_ds = build_eval_dataset(CompositionalDataset, hparams, eval_comp_final_json_path, hop)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=eval_ds,
        val_set=eval_ds
    )
    trainer.test_sequencial_compositional_ft(log=True, gap_num=gap_num, test_num=DEFAULT_TEST_NUM, artifact_options=make_artifact_options('LoRA-LLaVA'))

def test_MiniGPT4_one_lora_comp():
    '''
    LoRA baseline for MiniGPT4 compositional editing (single LoRA for both visual and textual)
    Uses CompositionalDataset (no Query Decomposition) + adversal2k.json
    '''
    hparams = LORAMultimodalHparams.from_hparams('hparams/LORA/minigpt4_compositional_one_lora.yaml')
    prepare_eval_hparams(hparams, preferred_gpus=[1], required_gpus=1)
    eval_ds = build_eval_dataset(CompositionalDataset, hparams, eval_comp_final_json_path, hop)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=eval_ds,
        val_set=eval_ds
    )
    trainer.test_sequencial_compositional_ft(log=True, gap_num=gap_num, test_num=DEFAULT_TEST_NUM, artifact_options=make_artifact_options('LoRA-MiniGPT4'))

####################### SERAC ##########################

def test_LLaVA_SERAC_comp():
    hparams = SERACMultimodalTrainingHparams.from_hparams('hparams/SERAC/llava.yaml')
    gpus = prepare_eval_hparams(hparams, preferred_gpus=[4, 5], required_gpus=2)
    eval_ds = build_eval_dataset(CompositionalCaptionDataset, hparams, eval_comp_final_json_path, hop)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=eval_ds,
        val_set=eval_ds
    )
    trainer.test_sequencial_multi_gpus(log=True, gap_num=gap_num, test_num=DEFAULT_TEST_NUM, comp=True, training=False, gpus=gpus, artifact_options=make_artifact_options('SERAC-LLaVA'))

def test_MiniGPT4_SERAC_comp():
    hparams = SERACMultimodalTrainingHparams.from_hparams('hparams/SERAC/minigpt4.yaml')
    gpus = prepare_eval_hparams(hparams, preferred_gpus=[6, 7], required_gpus=2)
    eval_ds = build_eval_dataset(CompositionalCaptionDataset, hparams, eval_comp_final_json_path, hop)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=eval_ds,
        val_set=eval_ds
    )
    trainer.test_sequencial_multi_gpus(log=True, gap_num=gap_num, test_num=DEFAULT_TEST_NUM, comp=True, training=False, gpus=gpus, artifact_options=make_artifact_options('SERAC-MiniGPT4'))

####################### WISE ##########################

def test_LLaVA_WISE_comp():
    hparams = WISEMultimodalHyperParams.from_hparams('hparams/WISE/llava.yaml')
    gpus = prepare_eval_hparams(hparams, preferred_gpus=[0, 1], required_gpus=2)
    eval_ds = build_eval_dataset(CompositionalCaptionDataset, hparams, eval_comp_final_json_path, hop)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=eval_ds,
        val_set=eval_ds
    )
    trainer.test_sequencial_multi_gpus(log=True, gap_num=gap_num, test_num=DEFAULT_TEST_NUM, comp=True, training=False, gpus=gpus, artifact_options=make_artifact_options('WISE-LLaVA'))

def test_MiniGPT4_WISE_comp():
    hparams = WISEMultimodalHyperParams.from_hparams('hparams/WISE/minigpt4.yaml')
    gpus = prepare_eval_hparams(hparams, preferred_gpus=[2, 3], required_gpus=2)
    eval_ds = build_eval_dataset(CompositionalCaptionDataset, hparams, eval_comp_final_json_path, hop)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=eval_ds,
        val_set=eval_ds
    )
    trainer.test_sequencial_multi_gpus(log=True, gap_num=gap_num, test_num=DEFAULT_TEST_NUM, comp=True, training=False, gpus=gpus, artifact_options=make_artifact_options('WISE-MiniGPT4'))

'''
OURS
'''

####################### Stage 2: Knowledge Connector Training  ##########################

def train_LLaVA_OURS_stage2():
    """Train Knowledge Connector for LLaVA with 70% RAG accuracy threshold (Stage 2)"""
    if gap_num == 0:
        hparams = LORAMultimodalHparams.from_hparams('hparams/OURS/stage2/llava_train_compositional_connector_rag_70.yaml')
        train_ds = CompositionalDataset_RAG_70(train_comp_final_json_path, config=hparams, hop=hop)
        trainer = MultimodalTrainer(
            config=hparams,
            train_set=train_ds,
            val_set=train_ds
        )
        trainer.test_sequencial_compositional_connector_attention_rag_70(log=True, gap_num=gap_num, test_num=500)
    else: 
        exit()

def train_MiniGPT4_OURS_stage2():
    """Train Knowledge Connector for MiniGPT4 with 50% RAG accuracy threshold (Stage 2)"""
    if gap_num == 0:
        hparams = LORAMultimodalHparams.from_hparams('hparams/OURS/stage2/minigpt4_train_compositional_connector_rag_50.yaml')
        train_ds = CompositionalDataset_RAG_50(train_comp_final_json_path, config=hparams, hop=hop)
        trainer = MultimodalTrainer(
            config=hparams,
            train_set=train_ds,
            val_set=train_ds
        )
        trainer.test_sequencial_compositional_connector_attention_rag_50(log=True, gap_num=gap_num, test_num=500)
    else: 
        exit()

####################### Test ##########################

def test_LLaVA_OURS_comp():
    hparams = OURSMultimodalHparams.from_hparams('hparams/OURS/llava.yaml')
    gpus = prepare_eval_hparams(hparams, preferred_gpus=[0, 1], required_gpus=2)
    eval_ds = build_eval_dataset(CompositionalCaptionDataset, hparams, eval_comp_final_json_path, hop)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=eval_ds,
        val_set=eval_ds
    )
    trainer.test_sequencial_multi_gpus(log=True, gap_num=gap_num, test_num=DEFAULT_TEST_NUM, comp=True, training=False, gpus=gpus, artifact_options=make_artifact_options('MemEIC-LLaVA'))

def test_MiniGPT4_OURS_comp():
    hparams = OURSMultimodalHparams.from_hparams('hparams/OURS/minigpt4.yaml')
    gpus = prepare_eval_hparams(hparams, preferred_gpus=[2, 3], required_gpus=2)
    eval_ds = build_eval_dataset(CompositionalCaptionDataset, hparams, eval_comp_final_json_path, hop)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=eval_ds,
        val_set=eval_ds
    )
    trainer.test_sequencial_multi_gpus(log=True, gap_num=gap_num, test_num=DEFAULT_TEST_NUM, comp=True, training=False, gpus=gpus, artifact_options=make_artifact_options('MemEIC-MiniGPT4'))


if __name__ == "__main__":
    function_name = sys.argv[1]

    #train_comp_json_path = 'datasets/train_comp.json'
    #eval_comp_new_json_path = 'datasets/eval_comp_0411.json' # 내가 검토후 데이터셋(Jiyun)

    train_comp_final_json_path = 'datasets/CCKEB_train.json'
    eval_comp_final_json_path = CUSTOM_EVAL_DATASET
    hop = DEFAULT_HOP

    if function_name not in globals() or not callable(globals()[function_name]):
        print(f"Error: Function '{function_name}' does not exist.")
        sys.exit(1)

    for gap_num in resolve_gap_schedule():
        globals()[function_name]()
