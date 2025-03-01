#!/bin/env python

import argparse
from tomlkit import parse, TOMLDocument
from pathlib import Path
import subprocess
import re
import shutil
from datetime import datetime
from enum import Enum

parser = argparse.ArgumentParser()

parser.add_argument("jobs_path", type=Path, help="location of config and dataset toml")
parser.add_argument(
	"-d",
	"--dry",
	action=argparse.BooleanOptionalAction,
	help="only show resulting name (training parameters)",
)

args = parser.parse_args()


class GROUPS(Enum):
	BASICS = "Basics"
	SAVE = "Save"
	SDv2 = "SDv2"
	NET = "Network_setup"
	LYCO = "LyCORIS"
	OPTIM = "Optimizer"
	LR = "Lr_scheduler"
	PRECISION = "Training_precision"
	IMPROV = "Further_improvement"
	ARB = "ARB"
	CAP = "Captions"
	ATTN = "Attention"
	AUG = "Data_augmentation"
	CACHE = "Cache_latents"
	SAMP = "Sampling_during_training"
	LOG = "Logging"
	REG = "Regularization"
	HUGGING = "Huggingface"
	DEBUG = "Debugging"
	DEPRECATED = "Deprecated"
	OTHER = "Others"


def prepare_basket(config: TOMLDocument, dataset: TOMLDocument):
	basket: dict[str:str] = dict()

	add_basename(basket, config)
	add_model(basket, config)
	add_optimizer(basket, config)
	add_scheduler(basket, config)
	add_ulr(basket, config)
	add_tlr(basket, config)
	add_batch(basket, config)
	add_epoch(basket, config)
	add_step(basket, config)
	add_network(basket, config)
	add_resolution(basket, config)
	add_snr(basket, config)
	add_debias(basket, config)
	add_ipng(basket, config)
	add_dataset(basket, dataset)

	return basket


def notation_normalize(n: float) -> str:
	"""Convert float to scientific notation fixed to an exponent.

	Example:
	1.6e-4 -> 16e-5
	"""
	if n == 1:
		return "1"

	c, e = f"{n:e}".split("e")
	c = c.rstrip("0").rstrip(".")
	e = e[1:].lstrip("0")  # slice to remove negative sign

	return f"{c}e{e}"


def li_str_to_dict(li: list[str]) -> dict:
	ret = dict()
	for i in li:
		k, v = i.split("=")
		ret[k] = v

	return ret


def get_basename(config: TOMLDocument) -> str:
	return config[GROUPS.SAVE.value]["output_name"]


def add_basename(basket: dict, config: TOMLDocument):
	# basename
	basename = config[GROUPS.SAVE.value]["output_name"]
	basket[basename] = ""


def add_model(basket: dict, config: TOMLDocument):
	# model
	basics = config.get(GROUPS.BASICS.value)
	model_name = Path(basics.get("pretrained_model_name_or_path")).stem
	if "noob" in model_name.lower():
		if "vpred" in model_name.lower():
			basket["m"] = "noobv"
		else:
			basket["m"] = "noob"


def add_optimizer(basket: dict, config: TOMLDocument):
	# optimizer
	optimizer = config.get(GROUPS.OPTIM.value)
	optimizer_name:str = optimizer.get("optimizer_type").lower()

	if "came" in optimizer_name:
		basket["o"] = "CAME"
		optimizer_args = li_str_to_dict(optimizer.get("optimizer_args"))
		update = optimizer_args.get('update_strategy')
		if update:
			basket['uC'] = ''

	elif "prodigy" in optimizer_name:
		basket["o"] = "Prodigy"

		optimizer_args = li_str_to_dict(optimizer.get("optimizer_args"))
		d_coef = optimizer_args.get("d_coef") or "1"
		basket["d"] = d_coef

	elif "fmarscrop" in optimizer_name:
		version = ''
		machina = ''
		mver = re.search(r'v\d', optimizer_name)
		mmach = re.search(r'exmachina', optimizer_name)

		args = ''

		if mver:
			version = f'{mver.group(0)}'

		if mmach:
			machina = f'xm'

			optimizer_args = li_str_to_dict(optimizer.get("optimizer_args"))

			update_strategy = optimizer_args.get('update_strategy')
			args += '' if update_strategy in [None, 'cautious'] else update_strategy

			if 'exmachina' in optimizer_name:
				eps_floor = optimizer_args.get('eps_floor')  # dynamic eps
				args += '' if eps_floor is None else f'epf{eps_floor}'

				moment_centralization = optimizer_args.get('moment_centralization') # def 0.0
				args += '' if not moment_centralization else f'mc{float(moment_centralization):g}'


		basket["o"] = 'fmc' + version + machina + (f'-{args}' if args else '')
	
	elif "compass" in optimizer_name:
		plus = ''
		if re.search(r'Plus', optimizer_name):
			plus = "p"

		basket["o"] = 'comp' + plus
	
	elif "schedulefree" in optimizer_name:
		sf_name = ''
		if "radam" in optimizer_name:
			sf_name = "radam"

		basket['o'] = 'SF' + sf_name

def add_scheduler(basket: dict, config: TOMLDocument):
	sch_conf = config.get(GROUPS.LR.value)
	sch_type: str | None = sch_conf.get("lr_scheduler_type")

	if sch_type:
		# TODO: all of torch built-in schedulers
		sch_args = li_str_to_dict(sch_conf.get("lr_scheduler_args"))
		if "rex" in sch_type.lower():
			# TODO: figure out what to grab for cycles
			basket["s"] = "rex"

			sch_suffix = ""
			if sch_args:
				d = sch_args.get("d")
				gamma = sch_args.get("gamma")
				cycles = sch_conf.get("lr_scheduler_num_cycles")
				warmup = sch_args.get("warmup_steps")
				sch_suffix += f'w{float(warmup):g}' if warmup and float(warmup) != 0 else ''
				sch_suffix += f"d{float(d) * 10:g}" if d and d != "0.9" else ""
				sch_suffix += (
					f"g{float(gamma) * 10:g}" if gamma and gamma != "0.9" else ""
				)
				sch_suffix += f"c{cycles}" if cycles != 1 else ""

				if sch_suffix:
					basket[sch_suffix] = ""
	else:
		sch = sch_conf.get("lr_scheduler")
		if "linear" in sch:
			basket["s"] = "lin"
		if "constant" in sch:
			basket["s"] = "cost"

		sch_suffix = ''
		warmup = sch_conf.get('lr_warmup_steps')
		if warmup:
			sch_suffix += f'w{float(warmup):g}' if warmup and float(warmup) != 0 else ''

		if sch_suffix:
			basket[sch_suffix] = ""


def add_ulr(basket: dict, config: TOMLDocument):
	# unet lr
	optimizer = config.get(GROUPS.OPTIM.value)
	ulr = optimizer.get("unet_lr")
	basket["u"] = notation_normalize(ulr)


def add_tlr(basket: dict, config: TOMLDocument):
	# te lr
	optimizer = config.get(GROUPS.OPTIM.value)
	network_setup = config.get("Network_setup")
	if not network_setup.get("network_train_unet_only"):
		tlr = optimizer.get("text_encoder_lr")
		basket["t"] = notation_normalize(tlr)


def add_batch(basket: dict, config: TOMLDocument):
	# batch
	optimizer = config.get(GROUPS.OPTIM.value)
	batch = optimizer.get("train_batch_size")
	basket["b"] = str(batch)


def add_epoch(basket: dict, config: TOMLDocument):
	# epoch
	basics = config.get(GROUPS.BASICS.value)
	epoch = basics.get("max_train_epochs")
	if epoch:
		basket["e"] = str(epoch)


def add_step(basket: dict, config: TOMLDocument):
	basics = config.get(GROUPS.BASICS.value)
	steps = basics.get("max_train_steps")
	if steps and "e" not in basket:
		basket["st"] = str(steps)


def add_network(basket: dict, config: TOMLDocument):
	# network
	lyco = config.get(GROUPS.LYCO.value)
	module = lyco.get("network_module")
	network_args = lyco.get("network_args")

	if network_args:
		network_args = li_str_to_dict(network_args)

	if "networks.lora" == module:
		basket["a"] = "lora"
		# I have been told there is little-to-no difference between training
		# vs inference lbw. So we will not care about this anymore
		# if parsed_network.get("down_lr_weight"):
		# weights = parsed_network["down_lr_weight"].split(",")
		# basket["wd"] = weights[0]

	elif "lycoris.kohya" == module:
		algo = network_args.get("algo")
		if algo == "lokr":
			if network_args.get("dora_wd"):
				algo = "d" + algo[1:]

			if network_args.get("wd_on_output"):
				algo += "r"

			basket["a"] = algo

			algo_suffix = ''
			factor = network_args.get("factor")
			full_matrix = network_args.get('full_matrix')

			algo_suffix += f"f{factor}" if factor else ""
			algo_suffix += "fm" if full_matrix else ""

			basket["".join(algo_suffix)] = ""

		elif algo == "locon":
			if network_args.get("dora_wd"):
				algo = "docon"
				if network_args.get("wd_on_output"):
					algo = "ddocon"
			basket["a"] = algo

	# dim/alpha
	#
	# dim/alpha are set by set by lokr factor, settings are irrelevant here?
	# if algo not in ["lokr", "dokr"]:
	network = config.get(GROUPS.NET.value)
	rank = ''
	rs_lora = network_args.get("rs_lora")

	rank += f'd{network.get("network_dim", 4)}' if not full_matrix else ''
	rank += f'a{network.get("network_alpha", 1)}'

	if network_args:
		rank += f'cd{network_args.get("conv_dim", 4)}' if not full_matrix else ''
		rank += f'ca{network_args.get("conv_alpha", 1)}'

		if rs_lora:
			rank += 'rs'

	basket[rank] = ""



def add_snr(basket: dict, config: TOMLDocument):
	# other improvements, else
	improvements = config.get(GROUPS.IMPROV.value)
	snr = improvements.get("min_snr_gamma")
	if snr:
		basket["snr"] = str(snr)


def add_ipng(basket: dict, config: TOMLDocument):
	improvements = config.get(GROUPS.IMPROV.value)
	ipng = improvements.get("ip_noise_gamma")
	if ipng:
		basket["ip"] = f"{ipng * 10:.1g}"


def add_debias(basket: dict, config: TOMLDocument):
	improvements = config.get(GROUPS.IMPROV.value)
	debiased = improvements.get("debiased_estimation_loss")
	if debiased:
		basket["db"] = ""


def add_resolution(basket: dict, config: TOMLDocument):
	# dataset and resolution
	basics = config.get(GROUPS.BASICS.value)
	training_resolution = basics.get("resolution")
	if training_resolution != "1024":
		basket["r"] = training_resolution


def add_dataset(basket: dict, dataset: TOMLDocument):
	dataset_general = dataset.get("general")
	datasets = dataset.get('datasets')
	dataset_names: list[str] = []
	for dataset in datasets:
		subsets = dataset.get('subsets')
		subset_names: list[str] = []
		for subset in subsets:
			dataset_name = Path(subset.get("image_dir")).stem
			repeats = subset.get('num_repeats')
			subset_names.append(f'{dataset_name}r{repeats}')

		dataset_names.append('_'.join(subset_names))

	datasets_subsets_joined = '-'.join(dataset_names)

	res = ''
	if dataset_general:
		dataset_resolution = dataset_general.get("resolution")
		if dataset_resolution != 1024:
			res = f"R{dataset_resolution}"
	
	basket[f"{f'-{res}' if res else ''}{datasets_subsets_joined}"] = ""


def is_find_lr(config: TOMLDocument):
	validation = config.get("Validation")
	if validation:
		return validation.get("is_find_lr")

	return False


def get_resume(config: TOMLDocument):
	network = config.get("Network_setup")
	if network:
		return network.get("resume")


def get_network_weights(config: TOMLDocument):
	network = config.get("Network_setup")
	if network:
		return network.get("network_weights")


def diff_basket_names(b1: str, b2: str):
	"""Return bakset elements of b1 not in b2"""
	b1_li = b1.split("-")
	b2_li = b2.split("-")

	for e2 in b2_li:
		if e2 in b1_li:
			b1_li.remove(e2)

	return "-".join(b1_li)


def natural_stem(p: Path) -> list:
	"""Chunk stem path by word-number for sorting.

	Python sorts by lexicograph, which is no good for humans. The general idea
	for this program is to watch a directory, and will choose the first time
	sorted naturally.

	This /works/ by chunking the stem into by string and number, put into a
	list. Numbers are forced to be ints so it gets sorted by it's numerical
	value rather than lexico, and string sorting works as expected.
	"""
	s = p.stem
	return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", s)]


def exists_handler(p: Path) -> Path:
	"""Return a renamed path if exists, otherwise return unchanged"""
	if not p.exists():
		return p

	name = p.stem
	ext = p.suffix

	number = 1
	if "_" in p.stem:
		try:
			name_split = name.split("_")
			number = int(name_split[-1])
			number += 1
			name = "_".join(name_split[:-1])
		except ValueError:
			pass

	return p.parent / f"{name}_{number}{ext}"


def main():
	jobs_path: Path = args.jobs_path
	failed_jobs: set = set()

	while True:
		jobs = sorted(list(jobs_path.iterdir()), key=natural_stem)
		job = None

		while jobs:
			_job = jobs.pop(0)
			if _job in failed_jobs:
				continue

			job = _job
			break

		if job is None:
			break

		config_file = job / "config.toml"
		dataset_file = job / "dataset.toml"

		with open(config_file, "r") as fp:
			config_file_content = fp.read()
			config = parse(config_file_content)

		with open(dataset_file, "r") as fp:
			dataset_file_content = fp.read()
			dataset = parse(dataset_file_content)

		basket = prepare_basket(config, dataset)
		basename = get_basename(config)
		name = "-".join(f"{k}{v}" for k, v in basket.items())
		name = name.replace(".", "_")  # convert decimals to _

		if is_find_lr(config):  # for temp trial runs for finding lr
			name = "find_lr-" + name

		# save to proper dir if resume, else normal
		#
		# a training set B is nested within a training set A, it should be
		# assumed that B is a continued training from A
		#
		# slight problem is that there my be a clash in naming scheme if
		# there are other models ran with the same parameters, but from
		# scratch, i guess we can just append name of the OG it's resuming
		# from. it'll look ugly but it'll work and guarantee a unique name...
		#
		# we append the difference in continuation parameters, much more elegant
		continue_from = get_network_weights(config) or get_resume(config)
		if continue_from:
			continue_from = Path(continue_from)
			continue_from_name = continue_from.stem

			if continue_from_name == "model":
				continue_from_name = continue_from.parent.stem

			# if continue from state folder
			continue_from_name = continue_from_name.replace("-state", "")

			diff_name = diff_basket_names(name, continue_from_name) or "same"

			if get_network_weights(config):
				name = f"{continue_from_name}-NETWEIGHTS-{diff_name}"
				output_dir = Path(continue_from).parent / f"NETWEIGHTS-{diff_name}"
			elif get_resume(config):
				name = f"{continue_from_name}-RESUME-{diff_name}"
				output_dir = Path(continue_from).parent / f"RESUME-{diff_name}"
		else:
			output_dir = Path(
				config[GROUPS.SAVE.value]["output_dir"].format(
					basename=basename, name=name
				)
			)

		output_file = output_dir / f"{name}.safetensors"

		# make sure output file does not already exist
		if output_file.exists():
			failed_jobs.add(job)
			print(
				f"the following output model already exists, skipping...\n{output_file}"
			)
			continue

		# confirm name
		print(
			f"training under the following name. make sure theses params are correct for this session:\n{name}\n"
		)

		# modify in-memory toml content from earlier read and do global sub
		new_config = config_file_content.format(
			name=name,
			basename=basename,
			datetime=datetime.now().strftime("%Y%m%d_%H%M%S"),
		)

		# if we're continuing from a previous training, overwrite output_dir
		# with our nested training session
		new_config = re.sub(
			"output_dir = .*", f'output_dir = "{output_dir}"', new_config
		)

		# change output name, which was basename, to name
		new_config = re.sub(
			re.escape(f'output_name = "{basename}"'),
			f'output_name = "{name}"',
			new_config,
		)

		new_dataset = dataset_file_content.format(basename=basename)

		# start training
		dry = args.dry
		if dry:
			failed_jobs.add(job)
			continue

		# write to runtime store
		runtime = Path("./backend/runtime_store/")
		runtime.mkdir(exist_ok=True)
		with open(runtime / "config.toml", "w") as fp:
			fp.write(new_config)

		with open(runtime / "dataset.toml", "w") as fp:
			fp.write(new_dataset)

		command = [
			"python",
			"./backend/sd_scripts/sdxl_train_network.py",
			"--dataset_config",
			"./backend/runtime_store/dataset.toml",
			"--config_file",
			"./backend/runtime_store/config.toml",
		]
		try:
			proc = subprocess.run(command, check=True)
			if proc.returncode == 0:
				archive_dir = job.parent.parent / "archive" / basename
				archive_dir.mkdir(parents=True, exist_ok=True)
				dst = archive_dir / name
				if dst.exists():
					shutil.move(dst, exists_handler(dst / "old"))
				shutil.move(job, archive_dir / name)  # also renames it appropriately
		except Exception as e:
			failed_jobs.add(job)
			print(e)


if __name__ == "__main__":
	main()
