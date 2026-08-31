"""Humanoid Olympics environment shared by the referee and local tools."""

from .course import EVENT_LABELS, EVENT_MAX_STEPS, EVENTS, TRACK_HALF_W, nominal_mu
from .scoring import instance_score, meet_score
from .sim import (ACT_DIM, OBS_DIM, STATE_DIM, InstanceParams, InvalidAction, OlympicsSim,
                  event_instances, instance_spec)

__all__ = ["ACT_DIM", "EVENT_LABELS", "EVENT_MAX_STEPS", "EVENTS", "OBS_DIM", "STATE_DIM",
           "InstanceParams", "InvalidAction", "OlympicsSim", "TRACK_HALF_W", "event_instances",
           "instance_score", "instance_spec", "meet_score", "nominal_mu"]
