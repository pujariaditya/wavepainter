import argparse
import os
import yaml

from wavepainter.runtime.os_utils import remove_file

global_print_hparams = True
hparams = {}


class Args:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            self.__setattr__(k, v)


def override_config(old_config: dict, new_config: dict):
    for k, v in new_config.items():
        if isinstance(v, dict) and k in old_config:
            override_config(old_config[k], new_config[k])
        else:
            old_config[k] = v


def set_hparams(config='', exp_name='', hparams_str='', dir='', print_hparams=True, global_hparams=True):
    if config == '' and exp_name == '':
        parser = argparse.ArgumentParser(description='')
        parser.add_argument('--config', type=str, default='',
                            help='location of the data corpus')
        parser.add_argument('--exp_name', type=str, default='', help='exp_name')
        # Default to the working directory, NOT ''. work_dir is built as
        # f'{args.dir}/checkpoints/{exp_name}', so an empty --dir resolves to
        # `/checkpoints/<name>` -- the filesystem root, which is not writable and
        # fails before anything is created. Override with --dir or
        # WAVEPAINTER_ROOT to put checkpoints elsewhere.
        parser.add_argument('--dir', type=str,
                            default=os.environ.get('WAVEPAINTER_ROOT', '.'),
                            help='root under which checkpoints/<exp_name> is written')
        parser.add_argument('-hp', '--hparams', type=str, default='',
                            help='location of the data corpus')
        parser.add_argument('--infer', action='store_true', help='infer')
        parser.add_argument('--validate', action='store_true', help='validate')
        parser.add_argument('--reset', action='store_true', help='reset hparams')
        parser.add_argument('--remove', action='store_true', help='remove old ckpt')
        parser.add_argument('--debug', action='store_true', help='debug')
        args, unknown = parser.parse_known_args()
        print("| Unknow hparams: ", unknown)
    else:
        args = Args(config=config, exp_name=exp_name, hparams=hparams_str, dir=dir,
                    infer=False, validate=False, reset=False, debug=False, remove=False)
    global hparams
    assert args.config != '' or args.exp_name != ''

    # `--dir` sets WHERE CHECKPOINTS GO. It used to also set where `--config` is
    # looked up (`args.dir + args.config`), and those want to be different things.
    #
    # Every agent must pass `--dir` so checkpoints land on the research drive, and
    # that drive's `egs` is a SYMLINK back to the shared seed. So an agent that
    # edited its own worktree config and launched with `--dir` silently trained the
    # UNEDITED shared copy: the run starts, the loss falls, and the arm under test
    # was never applied. Caught live by an agent whose printed hparams still showed
    # `training_mask_ratio: 0.8` after it had changed the value -- it noticed only
    # because it was queued for a GPU slot and had time to read the output.
    #
    # Resolve against the CURRENT DIRECTORY first, then fall back to `--dir`. The
    # worktree copy wins when both exist, which is what "my config" means. The
    # fallback is what `eval_ming.py` relies on: it passes
    # `checkpoints/<exp>/config.yaml`, which exists under `--dir` and not under the
    # grader's checkout, and must keep resolving there.
    config_path = args.config
    if args.config != '':
        local = os.path.abspath(args.config)
        # os.path.join, never concatenation: a `--dir` with no trailing slash,
        # say `<root>/wp`, plus config `checkpoints/x/config.yaml` otherwise
        # resolves to `<root>/wpcheckpoints/x/config.yaml`. Line 113 below
        # already joins properly, so the two disagreed, and this path only
        # worked when --dir was '.' or happened to end in a slash.
        via_dir = os.path.join(args.dir, args.config)
        if os.path.exists(local):
            config_path = local
        elif os.path.exists(via_dir):
            config_path = via_dir
        else:
            raise AssertionError(
                f"config {args.config!r} not found. Tried {local!r} (relative to the "
                f"current directory) and {via_dir!r} (relative to --dir). Run from "
                f"your worktree, or pass a path that exists under --dir.")

    config_chains = []
    loaded_config = set()

    def load_config(config_fn):
        # deep first inheritance and avoid the second visit of one node
        if not os.path.exists(config_fn):
            return {}
        with open(config_fn) as f:
            hparams_ = yaml.safe_load(f)
        loaded_config.add(config_fn)
        if 'base_config' in hparams_:
            ret_hparams = {}
            if not isinstance(hparams_['base_config'], list):
                hparams_['base_config'] = [hparams_['base_config']]
            for c in hparams_['base_config']:
                if c.startswith('.'):
                    c = f'{os.path.dirname(config_fn)}/{c}'
                    c = os.path.normpath(c)
                if c not in loaded_config:
                    override_config(ret_hparams, load_config(c))
            override_config(ret_hparams, hparams_)
        else:
            ret_hparams = hparams_
        config_chains.append(config_fn)
        return ret_hparams

    saved_hparams = {}
    args_work_dir = ''
    if args.exp_name != '':
        args_work_dir = os.path.join(args.dir, 'checkpoints', args.exp_name)
        ckpt_config_path = f'{args_work_dir}/config.yaml'
        if os.path.exists(ckpt_config_path):
            with open(ckpt_config_path) as f:
                saved_hparams_ = yaml.safe_load(f)
                if saved_hparams_ is not None:
                    saved_hparams.update(saved_hparams_)
    hparams_ = {}
    if args.config != '':
        hparams_.update(load_config(config_path))
    if not args.reset:
        hparams_.update(saved_hparams)
    hparams_['work_dir'] = args_work_dir

    # Support config overriding in command line. Support list type config overriding.
    # Examples: --hparams="a=1,b.c=2,d=[1 1 1]"
    if args.hparams != "":
        for new_hparam in args.hparams.split(","):
            k, v = new_hparam.split("=")
            v = v.strip("\'\" ")
            config_node = hparams_
            for k_ in k.split(".")[:-1]:
                config_node = config_node[k_]
            k = k.split(".")[-1]
            if v in ['True', 'False'] or type(config_node[k]) in [bool, list, dict]:
                if type(config_node[k]) == list:
                    v = v.replace(" ", ",")
                # Accept YAML booleans. The config files this parser overrides
                # are YAML, which writes `true`/`false`, so copying a key out of
                # egs/*.yaml and passing it back through --hparams is the
                # obvious thing to do -- and `eval("true")` raises
                # `NameError: name 'true' is not defined`, which blames the
                # value rather than the syntax and costs a run to diagnose.
                if isinstance(v, str) and v.strip().lower() in ('true', 'false'):
                    config_node[k] = v.strip().lower() == 'true'
                else:
                    config_node[k] = eval(v)
            else:
                config_node[k] = type(config_node[k])(v)
    if args_work_dir != '' and args.remove:
        answer = input("REMOVE old checkpoint? Y/N [Default: N]: ")
        if answer.lower() == "y":
            remove_file(args_work_dir)
    if args_work_dir != '' and (not os.path.exists(ckpt_config_path) or args.reset) and not args.infer:
        os.makedirs(hparams_['work_dir'], exist_ok=True)
        with open(ckpt_config_path, 'w') as f:
            yaml.safe_dump(hparams_, f)

    hparams_['infer'] = args.infer
    hparams_['debug'] = args.debug
    hparams_['validate'] = args.validate
    hparams_['exp_name'] = args.exp_name
    hparams_['exp_name'] = args.exp_name
    global global_print_hparams
    if global_hparams:
        hparams.clear()
        hparams.update(hparams_)
    if print_hparams and global_print_hparams and global_hparams:
        print('| Hparams chains: ', config_chains)
        print('| Hparams: ')
        for i, (k, v) in enumerate(sorted(hparams_.items())):
            print(f"\033[;33;m{k}\033[0m: {v}, ", end="\n" if i % 5 == 4 else "")
        print("")
        global_print_hparams = False
    return hparams_


def data_root(binary_data_dir=None):
    """Resolve `binary_data_dir` the same way everywhere.

    `directory` is the --dir prefix for CHECKPOINTS. Three call sites were
    prepending it to the data dir as well, so an absolute binary_data_dir
    became `/mnt/.../mnt/...` -- while a fourth read the hparam raw. Anything
    that needs the binarised pack goes through here.
    """
    d = binary_data_dir if binary_data_dir is not None else hparams['binary_data_dir']
    return d if os.path.isabs(d) else os.path.join(hparams.get('directory', ''), d)


def vocab_root():
    """Where `phone_set.json` / `word_set.json` / `spk_map.json` live.

    The SHIPPED `assets/vocab/` wins over `binary_data_dir`, and that ordering
    is what makes a released checkpoint scorable by anyone.

    A checkpoint's saved config records the absolute `binary_data_dir` of the
    machine that trained it. Resolving the vocabularies through it meant loading
    the released model required the training corpus to exist at that exact path
    -- so evaluation died in the task constructor, before a single frame was
    decoded, on a machine that had done nothing wrong.

    Kept separate from `data_root()` on purpose: that function also locates the
    binarised training pack, which must NOT fall back to a vocabulary
    directory. Only the token encoders come through here.

    These files must never be regenerated. Phone ids are positional, so a
    reordered `phone_set.json` silently mismatches the released checkpoint's
    embedding table -- wrong phone for every token, and nothing raises.
    """
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    shipped = os.path.join(repo_root, 'assets', 'vocab')
    if os.path.exists(os.path.join(shipped, 'phone_set.json')):
        return shipped
    return data_root()
