"""Merge completed SB3 run event streams into one display-only event file."""
import argparse,json
from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch.utils.tensorboard import SummaryWriter

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 a.output.mkdir(parents=True,exist_ok=False);events=[];files=sorted(a.source.glob('events.out.tfevents.*'))
 for f in files:
  acc=EventAccumulator(str(f),size_guidance={'scalars':0});acc.Reload()
  for tag in acc.Tags()['scalars']:
   events.extend((e.step,e.wall_time,tag,e.value) for e in acc.Scalars(tag))
 events.sort(key=lambda x:(x[0],x[1],x[2]))
 with SummaryWriter(str(a.output)) as writer:
  for step,wall,tag,value in events:writer.add_scalar(tag,value,step,walltime=wall)
 (a.output/'provenance.json').write_text(json.dumps({'source':str(a.source.resolve()),'files':[f.name for f in files],'scalar_events':len(events),'policy':'All scalar observations preserved, ordered by step and original walltime; no smoothing.'},indent=2))
 print(a.output,len(events))
if __name__=='__main__':main()
