1. Your adapted model is missing explicit state reset for spiking neurons, which can change behavior across forward passes.
Original code resets stateful modules at SeqSNN-RPE/SeqSNN/network/snn/ispikformer.py (line 425) with functional.reset_net(...) on encoder, embedding, and blocks before each forward. Your models/iSpikformer.py (line 148) does not reset any LIF state. With spikingjelly LIFNode(step_mode='m'), that can leak membrane state between batches/calls.

bunu check ele
