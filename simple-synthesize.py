"""ESpeak like synthesize to be used with aeneas."""
import argparse
import os
import time

import tensorflow as tf

from hparams import hparams
from infolog import log

from tacotron.synthesizer import Synthesizer
from tqdm import tqdm


def prepare_run(args):
	modified_hp = hparams.parse(args.hparams)
	os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

	run_name = args.name or args.tacotron_name or args.model
	taco_checkpoint = os.path.join('logs-' + run_name, 'taco_' + args.checkpoint)

	run_name = args.name or args.wavenet_name or args.model
	wave_checkpoint = os.path.join('logs-' + run_name, 'wave_' + args.checkpoint)
	return taco_checkpoint, wave_checkpoint, modified_hp


def get_sentences(args):
	if args.text_list != '':
		sentences = args.text_list.splitlines()
	elif args.text != '':
		sentences = [args.text.replace('\n', ' ')]
	else:
		sentences = hparams.sentences
	return sentences


def run_eval(args, checkpoint_path, output_dir, hparams, sentences):
	eval_dir = os.path.join(output_dir, 'eval')
	log_dir = os.path.join(output_dir, 'logs-eval')

	if args.model == 'Tacotron-2':
		assert os.path.normpath(eval_dir) == os.path.normpath(args.mels_dir)

	#Create output path if it doesn't exist
	os.makedirs(eval_dir, exist_ok=True)
	os.makedirs(log_dir, exist_ok=True)
	os.makedirs(os.path.join(log_dir, 'wavs'), exist_ok=True)
	os.makedirs(os.path.join(log_dir, 'plots'), exist_ok=True)

	synth = Synthesizer()
	synth.load(checkpoint_path, hparams)

	#Set inputs batch wise
	sentences = [sentences[i: i+hparams.tacotron_synthesis_batch_size] for i in range(0, len(sentences), hparams.tacotron_synthesis_batch_size)]

	#log('Starting Synthesis')
	with open(os.path.join(eval_dir, 'map.txt'), 'w') as file:
		for i, texts in enumerate(tqdm(sentences)):
			start = time.time()
			basenames = ['batch_{}_sentence_{}'.format(i, j) for j in range(len(texts))]
			mel_filenames, speaker_ids = synth.synthesize(texts, basenames, eval_dir, log_dir, None)

			for elems in zip(texts, mel_filenames, speaker_ids):
				file.write('|'.join([str(x) for x in elems]) + '\n')
	#log('synthesized mel spectrograms at {}'.format(eval_dir))
	return eval_dir


def tacotron_synthesize(args, hparams, checkpoint, sentences=None):
	output_dir = 'tacotron_' + args.output_dir

	try:
		checkpoint_path = tf.train.get_checkpoint_state(checkpoint).model_checkpoint_path
		log('loaded model at {}'.format(checkpoint_path))
	except:
		raise RuntimeError('Failed to load checkpoint at {}'.format(checkpoint))

	if hparams.tacotron_synthesis_batch_size < hparams.tacotron_num_gpus:
		raise ValueError('Defined synthesis batch size {} is smaller than minimum required {} (num_gpus)! Please verify your synthesis batch size choice.'.format(
			hparams.tacotron_synthesis_batch_size, hparams.tacotron_num_gpus))

	if hparams.tacotron_synthesis_batch_size % hparams.tacotron_num_gpus != 0:
		raise ValueError('Defined synthesis batch size {} is not a multiple of {} (num_gpus)! Please verify your synthesis batch size choice!'.format(
			hparams.tacotron_synthesis_batch_size, hparams.tacotron_num_gpus))

	return run_eval(args, checkpoint_path, output_dir, hparams, sentences)



def main():
	accepted_modes = ['eval']
	parser = argparse.ArgumentParser()
	parser.add_argument('--checkpoint', default='pretrained/', help='Path to model checkpoint')
	parser.add_argument('--hparams', default='',
		help='Hyperparameter overrides as a comma-separated list of name=value pairs')
	parser.add_argument('--name', help='Name of logging directory if the two models were trained together.')
	parser.add_argument('--tacotron_name', help='Name of logging directory of Tacotron. If trained separately')
	parser.add_argument('--wavenet_name', help='Name of logging directory of WaveNet. If trained separately')
	parser.add_argument('--model', default='Tacotron')
	parser.add_argument('--input_dir', default='training_data/', help='folder to contain inputs sentences/targets')
	parser.add_argument('--mels_dir', default='tacotron_output/eval/', help='folder to contain mels to synthesize audio from using the Wavenet')
	parser.add_argument('--output_dir', default='output/', help='folder to contain synthesized mel spectrograms')
	parser.add_argument('--mode', default='eval', help='mode of run: can be one of {}'.format(accepted_modes))
	parser.add_argument('--GTA', default='False', help='Ground truth aligned synthesis, defaults to True, only considered in synthesis mode')
	parser.add_argument('--text_list', default='', help='Sentences to separated by new lines. Valid if mode=eval')
	parser.add_argument("text", nargs='?', help='text to be synthesized')
	args = parser.parse_args()

	taco_checkpoint, wave_checkpoint, hparams = prepare_run(args)
	sentences = get_sentences(args)

	_ = tacotron_synthesize(args, hparams, taco_checkpoint, sentences)


if __name__ == '__main__':
	main()
