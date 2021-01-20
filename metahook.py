
import tensorflow as tf
from tensorflow.python.training.session_run_hook import SessionRunHook, SessionRunArgs
from tensorflow.python.training import training_util
from tensorflow.python.training.basic_session_run_hooks import SecondOrStepTimer
import os 
from tensorflow.python.client import timeline
import time 
class MetadataHook(SessionRunHook):
      def __init__(self, save_steps=None, save_secs=None, output_dir=""):
          self._output_tag = "blah-{}"
          self._output_dir = output_dir
          self._timer = SecondOrStepTimer(every_secs=save_secs,
                                          every_steps=save_steps)
          self._atomic_counter = 0
          self.start_time = None 
      def begin(self):
          self._next_step = None
          self._global_step_tensor = training_util.get_global_step()
          self._writer = tf.summary.FileWriter(self._output_dir,
                                               tf.get_default_graph())
          
          if self._global_step_tensor is None:
              raise RuntimeError(
                  "Global step should be created to use ProfilerHook.")
  
      def before_run(self, run_context):
          self._request_summary = (self._next_step is None
                                   or self._timer.should_trigger_for_step(
                                       self._next_step))
          requests = {}#{"global_step": self._global_step_tensor}
          opts = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
          self.start_time = time.time()
          tf.logging.info(f'Before Run: {time.time()}')
          return SessionRunArgs(requests, options=opts)
  
      def after_run(self, run_context, run_values):
          tf.logging.info(f'Inference Time: {time.time() - self.start_time}')
          global_step = self._atomic_counter + 1
          self._atomic_counter = self._atomic_counter + 1
        #   if self._request_summary:
        #       tf.logging.error(f'global step is {global_step}, atomic counter is {self._atomic_counter}')
        #     #   fetched_timeline = timeline.Timeline(run_values.run_metadata.step_stats)
        #     #   chrome_trace = fetched_timeline.generate_chrome_trace_format()
        #     #   with open(os.path.join(self._output_dir, f'timeline_{global_step}.json'), 'w') as f:
        #     #   with tf.gfile.GFile(os.path.join(self._output_dir, f'timeline_{global_step}.json'), 'w') as f:
        #     #     f.write(chrome_trace)
  
        #       self._writer.add_run_metadata(run_values.run_metadata,
        #                                     self._output_tag.format(global_step))
        #       self._writer.flush()
          self._next_step = global_step + 1
  
      def end(self, session):
          self._writer.close()