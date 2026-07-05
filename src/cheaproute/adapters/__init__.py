"""Transport adapters. The scoring environment's exact interface is unknown
until kickoff, so three transports are prebuilt and selected via config/env
(CHEAPROUTE_ADAPTER = stdio | http | cli). Only this layer should need to
change on kickoff day."""
