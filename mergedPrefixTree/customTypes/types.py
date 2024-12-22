from dataclasses import dataclass
from pm4py.objects.trie.obj import Trie
from damgard_jurik_utils.damgard_jurik_local.crypto import PublicKey


# types are:
#  organization identifier: orgId: int
#  activity: act: string
#  candidate identifier : candId: (organization identifier, int)

@dataclass(unsafe_hash=True)
class CandId:
    orgId: str
    id: int
    weight: int | None = 0

OrgId = str
Act = str

class AnomTrie(Trie):
    def __init__(self, *args, candId=None, **kwargs):
        super().__init__(*args, **kwargs)  # Call the parent class's constructor
        self.candId: CandId = candId  # Add the new property
        if candId is not None:
            self.label = str(self.candId.orgId) + " " + str(self.candId.id)
        self.mask = None
        self.weight: int = 0
        self.weightFrac: float = 0.0
        self.act = None
        self.children: list[AnomTrie] = []
        self.sensitiveActivities: dict[tuple[OrgId, Act], int] = None

class OrgTrie(Trie):
    def __init__(self, *args, activity=None, **kwargs):
        super().__init__(*args, **kwargs)  # Call the parent class's constructor
        self.act: Act = activity  # Add the new property
        if activity is not None:
            self.label = self.act
        self.timestamps: list[ActivityTimestamp] = []
        self.weight = 0

@dataclass
class Message:
    data: any
    action: str | None = None
    id: str | None = None
    sender: OrgId = None

@dataclass
class OrgPort:
    orgId: OrgId
    port: int

@dataclass
class ActivityTimestamp:
    act: Act
    time: int

@dataclass
class PartialEncTime:
    encTime: int
    partialEncTime: int
    partialEncI: int

@dataclass
class EncCandId:
    encOrgId: int
    encId: int

@dataclass
class PartialEncCandId:
    encOrgId: int
    encId: int
    partialEncId: int
    partialEncOrgId: int
    partialEncI: int

@dataclass
class CandidateTimestamp:
    candId: CandId | EncCandId | PartialEncCandId
    time: int | PartialEncTime

@dataclass
class PartialEncInt:
    encInt: int
    partialEncInt: int
    partialEncI: int
