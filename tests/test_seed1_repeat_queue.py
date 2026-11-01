"""Protect independent training seeds, fixed testing conditions and task scope."""
import ast
import importlib.util
from pathlib import Path

from omegaconf import OmegaConf

MODULE=Path(__file__).resolve().parents[1]/'scripts/queue_t_roundabout_seed1.py'
spec=importlib.util.spec_from_file_location('seed1_queue',MODULE)
queue=importlib.util.module_from_spec(spec);spec.loader.exec_module(queue)


def test_resolved_environment_seed_changes_with_training_seed():
    reference=OmegaConf.create(dict(seed=0,env=dict(seed=0,steps=500000,start_seed=0,eval_start_seed=10000),
        logdir='/old',model=dict(context=dict(enabled=True,swap_weight=1.)),batch_size=16))
    config,changes=queue.seed_config(reference,Path('/new'))
    assert config.seed==config.env.seed==1
    assert reference.seed==reference.env.seed==0
    assert set(changes)=={'seed','env.seed','env.start_seed','env.eval_start_seed','logdir','hydra.run.dir'}
    assert config.model==reference.model and config.batch_size==16
    assert config.env.seed+config.env.start_seed==0
    assert config.env.seed+config.env.eval_start_seed==10000
    assert config.env.seed+config.env.eval_start_seed+20-1<queue.TESTS[0][1]


def test_both_training_groups_exclude_dreamerv3_context():
    assert len(queue.METHODS)==5
    assert set(queue.METHODS)=={'ppo','sac','dreamerv3','r2dreamer','r2dreamer_context'}
    assert [x[1] for x in queue.TESTS]==[10020,10320,10420]


def test_sb3_test_rng_override_does_not_modify_training_rng():
    source=queue.ROOT/'logdir/tintersection_baselines_queue_20261022/source/scripts/train_metadrive_sb3.py'
    original=source.read_text();patched=queue.patch_sb3_tests(original)
    ast.parse(patched)
    assert "seed=int(c.seed)" in patched
    assert 'ec.seed=a.test_environment_seed;ec.eval_start_seed=start-int(ec.seed)' in patched
    assert original[:original.index("    p.add_argument('--steps'")]==patched[:patched.index("    p.add_argument('--steps'")]


def test_dreamer_test_rng_override_preserves_training_seed():
    original=(queue.ROOT/'scripts/evaluate_metadrive.py').read_text()
    patched=queue.patch_dreamer_tests(original);ast.parse(patched)
    assert 'config.env.seed=args.environment_seed' in patched
    assert 'training_seed=int(config.seed)' in patched
    assert 'config.seed=args.environment_seed' not in patched



import json
import tempfile
import unittest
from unittest.mock import patch


class RoundaboutResumeTests(unittest.TestCase):
    def prepare_queue(self, directory):
        jobs=[]; records={}
        for group in ('tintersection','roundabout_B'):
            for method in queue.METHODS:
                name=group+'_'+method
                jobs.append(dict(name=name,group=group,method=method,logdir=str(directory/name),
                    source=str(directory),command=['mock-trainer',name],evaluations=[]))
                records[name]=dict(state='completed' if group=='tintersection' else 'paused')
        queue.write(directory/'plan.json',dict(jobs=jobs,sources={},artifacts_sha256={}))
        queue.write(directory/'status.json',dict(phase='paused',jobs=records))
        return records

    def test_only_five_paused_roundabout_jobs_run(self):
        launched=[]
        class Child:
            pid=123456789
            def __init__(self,command,**kwargs): launched.append(command[1])
            def wait(self): return 0
            def poll(self): return 0
        with tempfile.TemporaryDirectory() as name:
            directory=Path(name); records=self.prepare_queue(directory)
            original={k:v for k,v in records.items() if k.startswith('tintersection_')}
            with patch.object(queue.subprocess,'Popen',Child), \
                 patch.object(queue,'validate_training',lambda job:Path(job['logdir'])/'best.mock'), \
                 patch.object(queue,'retain_evaluation',lambda *args:None), \
                 patch.object(queue.signal,'signal',lambda *args:None):
                queue.run(directory,resume_roundabout=True)
            state=json.loads((directory/'status.json').read_text())
            self.assertEqual(launched,['roundabout_B_'+method for method in queue.METHODS])
            self.assertEqual(state['phase'],'completed')
            self.assertEqual({k:v for k,v in state['jobs'].items() if k.startswith('tintersection_')},original)
            with self.assertRaisesRegex(RuntimeError,'paused queue'):
                queue.run(directory,resume_roundabout=True)

    def test_existing_partial_output_prevents_resume(self):
        with tempfile.TemporaryDirectory() as name:
            directory=Path(name); self.prepare_queue(directory)
            (directory/'roundabout_B_ppo').mkdir()
            with patch.object(queue.subprocess,'Popen',side_effect=AssertionError('Must not launch')):
                with self.assertRaisesRegex(RuntimeError,'untouched paused jobs'):
                    queue.run(directory,resume_roundabout=True)


class TrainingAdoptionTests(unittest.TestCase):
    def test_adopted_process_requires_matching_start_time_and_command(self):
        path=MODULE.parent/'queue_rhc_followup.py'
        spec=importlib.util.spec_from_file_location('followup_adoption_test',path)
        followup=importlib.util.module_from_spec(spec);spec.loader.exec_module(followup)
        fields=['0']*20;fields[0]='R';fields[19]='1234'
        stat='42 (python) '+' '.join(fields)
        handoff=dict(training_pid=42,training_start_ticks='1234',training_command=['python','train.py'])
        with patch.object(Path,'read_text',return_value=stat), \
             patch.object(Path,'read_bytes',return_value=b'python\0train.py\0'):
            self.assertTrue(followup.adopted_training_active(handoff))
            self.assertFalse(followup.adopted_training_active(dict(handoff,training_start_ticks='different')))
            with self.assertRaisesRegex(RuntimeError,'identity changed'):
                followup.adopted_training_active(dict(handoff,training_command=['other-process']))
        fields[0]='Z'
        with patch.object(Path,'read_text',return_value='42 (python) '+' '.join(fields)):
            self.assertFalse(followup.adopted_training_active(handoff))
        with patch.object(Path,'read_text',side_effect=FileNotFoundError):
            self.assertFalse(followup.adopted_training_active(handoff))

    def test_completed_h1_continuation_runs_seed0_then_roundabout(self):
        path=MODULE.parent/'queue_rhc_followup.py'
        spec=importlib.util.spec_from_file_location('followup_continue_test',path)
        followup=importlib.util.module_from_spec(spec);spec.loader.exec_module(followup)
        with tempfile.TemporaryDirectory() as name:
            base=Path(name);h1=base/'h1';retry=base/'retry';roundabout=base/'roundabout'
            for p in (h1,retry,roundabout):p.mkdir()
            followup.save(h1/'status.json',dict(phase='failed',h1_comparison='completed',
                completed_stages=['h1_training','h1_test_ID','h1_test_S','h1_test_XL']))
            followup.save(retry/'status.json',dict(phase='prepared'))
            followup.save(retry/'schedule.json',dict(control_sha256={}))
            followup.save(h1/'comparison.json',{})
            commands=[]
            class Child:
                pid=123456789
                def __init__(self,command,**kwargs):
                    commands.append(command)
                    if command==['mock-roundabout']:followup.save(roundabout/'status.json',dict(phase='completed'))
                    elif command[3]=='report':followup.save(retry/'status.json',dict(phase='completed'))
                def wait(self):return 0
                def poll(self):return 0
            plan=dict(resume_roundabout=['mock-roundabout'],roundabout=str(roundabout))
            with patch.object(followup,'verify',return_value=plan), \
                 patch.object(followup,'completed_episodes',return_value=([],{})), \
                 patch.object(followup.queue,'validate_training',return_value=h1/'best.pt'), \
                 patch.object(followup.subprocess,'Popen',Child), \
                 patch.object(followup.signal,'signal',lambda *args:None):
                followup.continue_seed0(h1,retry)
                self.assertEqual([c[3] for c in commands[:-1]],['train','evaluate','report'])
                self.assertEqual(commands[-1],['mock-roundabout'])
                self.assertEqual(followup.read(h1/'status.json')['phase'],'completed')
                with self.assertRaisesRegex(RuntimeError,'failed downstream queue'):
                    followup.continue_seed0(h1,retry)


class ResultNamespaceTests(unittest.TestCase):
    def test_rerun_keeps_original_seed1_table_and_source_data(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);table=root/'iv_output/table';table.mkdir(parents=True)
            old=table/'tintersection_seed1_test_results.csv';old.write_text('original result, must remain unchanged\n')
            output=root/'repeat';output.mkdir();run=output/'tintersection_ppo';run.mkdir()
            checkpoint=run/'best.zip';checkpoint.write_bytes(b'test checkpoint')
            config=output/'config.yaml';config.write_text('seed: 1\n')
            metrics={k:0.0 for k in ('success','crash','out_of_road','timeout','reward','route_completion','mean_speed_km_h','steps')}
            for split,start,_ in queue.TESTS:
                folder=run/'evaluation'/split;folder.mkdir(parents=True)
                queue.write(folder/'summary.json',dict(summary=metrics))
                (folder/'episodes.jsonl').write_text(''.join(json.dumps(dict(seed=s))+'\n' for s in range(start,start+100)))
            job=dict(name='tintersection_ppo',group='tintersection',method='ppo',logdir=str(run),
                config=str(config),source='/frozen',reference='/seed0',result_namespace='new_seed1_repeat',training_seed=2)
            with patch.object(queue,'ROOT',root):queue.retain_evaluation(output,job,checkpoint)
            self.assertEqual(old.read_text(),'original result, must remain unchanged\n')
            self.assertTrue((table/'new_seed1_repeat_tintersection_test_results.csv').exists())
            self.assertTrue((table/'source_data/new_seed1_repeat/tintersection_ppo/provenance.json').exists())
            provenance=json.loads((table/'source_data/new_seed1_repeat/tintersection_ppo/provenance.json').read_text())
            self.assertEqual(provenance['job']['training_seed'],2)
            round_job=dict(job,name='roundabout_B_ppo',group='roundabout_B')
            with patch.object(queue,'ROOT',root):queue.retain_evaluation(output,round_job,checkpoint)
            self.assertTrue((table/'new_seed1_repeat_roundabout_B_test_results.csv').exists())
            self.assertTrue((table/'new_seed1_repeat_tintersection_test_results.csv').exists())
            self.assertEqual(old.read_text(),'original result, must remain unchanged\n')
