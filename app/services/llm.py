"""
LLM provider abstraction layer.

Supports three modes:
  - mock:              Deterministic, heuristic-based responses. No API key needed.
  - openai:            OpenAI API via the official SDK.
  - openai_compatible: Any OpenAI-compatible endpoint (e.g. Ollama, Azure OpenAI).

The application always falls back to mock mode if an API call fails, ensuring
the demo is always runnable.
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from app.core.config import get_settings
from app.core.exceptions import LLMError, LLMParseError
from app.core.logging import get_logger
from app.core.metrics import get_registry

logger = get_logger(__name__)


@dataclass
class Message:
    """A single message in a chat conversation."""

    role: str  # "system" | "user" | "assistant"
    content: str


@dataclass
class LLMResponse:
    """Response from an LLM call."""

    content: str
    model: str
    provider: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    is_mock: bool = False


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def generate(
        self,
        messages: list[Message],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Generate a response from the model."""


# ---------------------------------------------------------------------------
# Mock LLM Provider
# ---------------------------------------------------------------------------
class MockLLMProvider(LLMProvider):
    """
    Deterministic mock LLM for development and testing.

    Detects the prompt type from the system message and returns a
    structured, realistic-looking response without any API calls.
    """

    def generate(
        self,
        messages: list[Message],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        import time

        start = time.perf_counter()
        system_content = next((m.content for m in messages if m.role == "system"), "").lower()
        user_content = next((m.content for m in messages if m.role == "user"), "")

        # Order matters: more specific checks first
        if (
            "intent extraction" in system_content
            or "parse this travel request" in user_content.lower()
        ):
            content = self._mock_intent(user_content)
        elif "critique" in system_content or "quality assurance" in system_content:
            content = self._mock_critique(user_content)
        elif (
            "friendly and knowledgeable travel assistant" in system_content
            or "synthesize" in system_content
        ):
            content = self._mock_final_answer(user_content)
        elif "itinerary" in system_content or "day-by-day" in system_content:
            content = self._mock_itinerary(user_content)
        else:
            content = self._mock_final_answer(user_content)

        elapsed_ms = (time.perf_counter() - start) * 1000
        get_registry().counter("llm.mock_calls").inc()

        return LLMResponse(
            content=content,
            model="mock-v1",
            provider="mock",
            input_tokens=len(" ".join(m.content for m in messages).split()),
            output_tokens=len(content.split()),
            latency_ms=elapsed_ms,
            is_mock=True,
        )

    def _mock_intent(self, user_content: str) -> str:
        """Parse the user query with lightweight heuristics."""
        text = user_content.lower()

        # City detection
        city = "Amsterdam"
        city_map = {
            "amsterdam": "Amsterdam",
            "paris": "Paris",
            "barcelona": "Barcelona",
            "rome": "Rome",
            "lisbon": "Lisbon",
        }
        for key, val in city_map.items():
            if key in text:
                city = val
                break

        # Days detection
        days = 3
        day_patterns = [
            r"(\d+)[- ]day",
            r"for (\d+) days?",
            r"(\d+) nights?",
        ]
        for pat in day_patterns:
            m = re.search(pat, text)
            if m:
                days = int(m.group(1))
                break

        # Travelers detection
        travelers = 2
        traveler_patterns = [
            r"for (\d+) (people|persons|travelers|travellers)",
            r"group of (\d+)",
            r"family of (\d+)",
        ]
        for pat in traveler_patterns:
            m = re.search(pat, text)
            if m:
                travelers = int(m.group(1))
                break
        if "solo" in text or "alone" in text or "myself" in text:
            travelers = 1
        if "couple" in text or "two of us" in text or "partner" in text or "honeymoon" in text:
            travelers = 2

        # Budget detection
        budget_level = "mid"
        if any(w in text for w in ["luxury", "five star", "5 star", "high-end", "splurge"]):
            budget_level = "luxury"
        elif any(w in text for w in ["upper mid", "upper-mid", "nice but not luxury"]):
            budget_level = "upper_mid"
        elif any(w in text for w in ["budget", "cheap", "backpack", "hostel", "affordable"]):
            budget_level = "budget"

        # Interests detection
        interests = []
        interest_map = {
            "museum": "museums",
            "art": "art",
            "history": "history",
            "food": "food",
            "restaurant": "food",
            "nightlife": "nightlife",
            "party": "nightlife",
            "club": "nightlife",
            "beach": "beach",
            "nature": "nature",
            "cycling": "cycling",
            "bike": "cycling",
            "canal": "canals",
            "architecture": "architecture",
            "shopping": "shopping",
            "fado": "fado",
            "jazz": "jazz",
            "wine": "wine",
            "market": "markets",
        }
        for keyword, interest in interest_map.items():
            if keyword in text and interest not in interests:
                interests.append(interest)

        # Travel style
        style = "cultural"
        if any(w in text for w in ["romantic", "honeymoon", "couple"]):
            style = "romantic"
        elif any(w in text for w in ["family", "kids", "children"]):
            style = "family"
        elif any(w in text for w in ["business", "conference", "work"]):
            style = "business"
        elif any(w in text for w in ["nightlife", "party", "club"]):
            style = "nightlife"
        elif any(w in text for w in ["relax", "chill", "slow"]):
            style = "relaxation"

        # Special requests
        special_requests = []
        if "day trip" in text or "day-trip" in text:
            special_requests.append("day trip outside the city")
        if "canal cruise" in text:
            special_requests.append("canal cruise")
        if "private tour" in text:
            special_requests.append("private guided tour")
        if "vegetarian" in text or "vegan" in text:
            special_requests.append("vegetarian/vegan food options")

        return json.dumps(
            {
                "city": city,
                "days": days,
                "travelers": travelers,
                "budget_level": budget_level,
                "interests": interests or ["sightseeing", "food"],
                "travel_style": style,
                "accommodation_preferences": [
                    "central location",
                    "near public transport",
                ],
                "special_requests": special_requests,
                "start_date": None,
                "end_date": None,
            }
        )

    def _mock_itinerary(self, user_content: str) -> str:
        """Generate a template-based but realistic itinerary."""
        # Extract key fields from the prompt
        city_match = re.search(r"in (\w+)\.", user_content)
        city = city_match.group(1) if city_match else "Amsterdam"

        days_match = re.search(r"(\d+)-day itinerary", user_content)
        days = int(days_match.group(1)) if days_match else 3

        city_templates: dict[str, dict[str, Any]] = {
            "Amsterdam": {
                "trip_name": "Amsterdam Canals & Culture",
                "overview": "A beautifully curated Amsterdam experience blending iconic canal scenery, world-class museums, and the city's vibrant neighbourhood life.",
                "days": [
                    {
                        "day": 1,
                        "theme": "Arrival & Canal Heart",
                        "morning": "Settle into your hotel and take a morning stroll through the Jordaan neighbourhood, grabbing breakfast at a traditional Dutch bruin café.",
                        "afternoon": "Visit the Rijksmuseum (book ahead) to see Rembrandt's Night Watch and Vermeer's Milkmaid. Allow 2-3 hours.",
                        "evening": "Take a candlelit evening canal cruise, then dine at Buffet van Odette in Jordaan for seasonal Dutch cuisine.",
                        "transport_tip": "Most museums are walkable from the city centre. Use the OV-chipkaart for tram rides.",
                        "estimated_daily_budget": "€90-130 per person",
                    },
                    {
                        "day": 2,
                        "theme": "Art, History & Waterways",
                        "morning": "Beat the queues at the Van Gogh Museum with a pre-booked timed entry. The permanent collection takes about 2 hours.",
                        "afternoon": "Visit the Anne Frank House (timed entry essential) in the early afternoon, then wander the Prinsengracht canal belt.",
                        "evening": "Dinner in De Pijp at one of the neighbourhood's excellent global restaurants, followed by drinks at Brouwerij 't IJ in the windmill.",
                        "transport_tip": "Tram 2 or 12 connects the Museum Quarter to De Pijp quickly.",
                        "estimated_daily_budget": "€80-120 per person",
                    },
                    {
                        "day": 3,
                        "theme": "Day Trip & Local Life",
                        "morning": "Take the train to Keukenhof (if March-May) or the windmills at Zaanse Schans. Both are under 30 minutes from Centraal.",
                        "afternoon": "Return to Amsterdam and explore the Albert Cuyp market in De Pijp for local cheeses, stroopwafels, and herring.",
                        "evening": "Farewell dinner at De Kas in Frankendael park — Amsterdam's finest farm-to-table experience.",
                        "transport_tip": "Intercity trains depart from Amsterdam Centraal every 15 minutes to Schiphol for Keukenhof connections.",
                        "estimated_daily_budget": "€100-150 per person",
                    },
                ],
                "total_estimated_cost": "€270-400 per person (excluding flights & hotel)",
                "practical_tips": [
                    "Book museum tickets well in advance — the Rijksmuseum, Van Gogh Museum, and Anne Frank House all sell out weeks ahead.",
                    "Rent a bike for a half-day to truly experience Amsterdam like a local. OV Fiets at Centraal is reliable.",
                    "The GVB tram network is excellent. A 24-hour pass (€9.50) offers unlimited rides.",
                ],
                "best_time_to_visit": "April-May (tulip season, pleasant weather) or September (quieter, still warm).",
            },
            "Paris": {
                "trip_name": "Paris: Art, Gastronomy & Romance",
                "overview": "An immersive Paris journey through iconic landmarks, world-class museums, and the intimate neighbourhood life that makes the city endlessly compelling.",
                "days": [
                    {
                        "day": 1,
                        "theme": "Left Bank & Impressionist Paris",
                        "morning": "Start at the Musée d'Orsay for Monet, Degas, and Renoir — arrive at opening to avoid queues. Allow 2.5 hours.",
                        "afternoon": "Wander the Luxembourg Gardens and Saint-Germain-des-Prés, stopping at Café de Flore for the obligatory café au lait.",
                        "evening": "Dinner at Septime in the 11th — book well in advance. One of Paris's best modern bistros.",
                        "transport_tip": "The Saint-Germain area is very walkable. Métro line 4 connects to Montparnasse if needed.",
                        "estimated_daily_budget": "€120-180 per person",
                    },
                    {
                        "day": 2,
                        "theme": "The Louvre & Le Marais",
                        "morning": "The Louvre opens at 9am — arrive 30 minutes early. Focus on the Denon wing: Mona Lisa, Winged Victory, Venus de Milo.",
                        "afternoon": "Lunch at L'As du Fallafel in the Marais, then explore Place des Vosges and the Picasso Museum.",
                        "evening": "Aperitivo at Au Passage wine bar, then an evening Seine cruise for the illuminated Eiffel Tower.",
                        "transport_tip": "Métro line 1 connects the Louvre to the Marais (Saint-Paul station) in 10 minutes.",
                        "estimated_daily_budget": "€110-160 per person",
                    },
                    {
                        "day": 3,
                        "theme": "Montmartre & Sacré-Cœur",
                        "morning": "Climb to Montmartre at sunrise for the best light and fewest crowds. Sacré-Cœur, the vineyard, and the original Place du Tertre.",
                        "afternoon": "Versailles day trip by RER B (40 min). Allow a full afternoon for the palace and gardens.",
                        "evening": "Return to Paris for dinner in a classic Left Bank brasserie. Bouillon Chartier for authentic Belle Époque atmosphere.",
                        "transport_tip": "RER C from Champ de Mars to Versailles takes 35 minutes and runs every 15 minutes.",
                        "estimated_daily_budget": "€130-200 per person",
                    },
                ],
                "total_estimated_cost": "€360-540 per person (excluding flights & hotel)",
                "practical_tips": [
                    "Paris Museum Pass (2 or 4 days) covers the Louvre, Orsay, Versailles, and 60+ sites. Excellent value.",
                    "Always greet with 'Bonjour' — it makes a genuine difference to how you're received.",
                    "Avoid tourist trap restaurants near Notre-Dame. Walk one street back from any major sight for authentic local options.",
                ],
                "best_time_to_visit": "April-June or September-October for pleasant weather and manageable crowds.",
            },
            "Barcelona": {
                "trip_name": "Barcelona: Gaudí, Gastronomy & the Med",
                "overview": "Barcelona dazzles with its unique Modernista architecture, world-renowned food scene, golden Mediterranean coastline, and neighbourhoods each with their own distinct character.",
                "days": [
                    {
                        "day": 1,
                        "theme": "Gaudí's City",
                        "morning": "Begin at the Sagrada Família — book timed-entry tickets with tower access months in advance. Allow 2.5 hours for the full experience.",
                        "afternoon": "Walk or take the metro to Park Güell. The monumental zone (timed entry) offers mosaic terraces and panoramic Barcelona views.",
                        "evening": "Aperitivo in El Born — start at Bar del Pla for patatas bravas and vermut, then explore the neighbourhood's pintxos bars.",
                        "transport_tip": "Metro L2 (Sagrada Família) and L3 (Vallcarca for Park Güell) make both sites easy to reach.",
                        "estimated_daily_budget": "€80-120 per person",
                    },
                    {
                        "day": 2,
                        "theme": "Gothic Lanes & Market Life",
                        "morning": "Breakfast at Bar Pinotxo in La Boqueria (arrive by 9am). Then walk the Gothic Quarter: Barcelona Cathedral, Plaça del Rei, Roman ruins.",
                        "afternoon": "Barceloneta beach for a couple of hours, followed by seafood lunch at La Cova Fumada — birthplace of the bomba.",
                        "evening": "Tapas crawl through El Born: Cervecería Catalana, Bar del Pla, and finish at El Xampanyet with cava.",
                        "transport_tip": "The Gothic Quarter is best on foot. Barceloneta is a 20-minute walk from the Gothic Quarter along the waterfront.",
                        "estimated_daily_budget": "€70-100 per person",
                    },
                    {
                        "day": 3,
                        "theme": "Montjuïc & Gràcia",
                        "morning": "Take the cable car or funicular to Montjuïc. MNAC for Romanesque art, then the gardens and Olympic stadium.",
                        "afternoon": "Descend to the Gràcia neighbourhood for lunch and a wander through its lively plazas (Plaça del Sol, Plaça de la Vila de Gràcia).",
                        "evening": "Dinner at Tickets (reserve weeks ahead) for Albert Adrià's avant-garde tapas — one of Barcelona's most memorable meals.",
                        "transport_tip": "The Montjuïc cable car runs from Barceloneta beach. Alternatively, Metro L3 to Paral·lel, then funicular.",
                        "estimated_daily_budget": "€90-140 per person",
                    },
                ],
                "total_estimated_cost": "€240-360 per person (excluding flights & hotel)",
                "practical_tips": [
                    "Book Sagrada Família and Casa Batlló/Casa Milà months in advance — they sell out. No exceptions.",
                    "Eat lunch between 2-4pm and dinner after 9pm like the locals. You'll get better tables and authentic atmosphere.",
                    "Pickpockets are active on Las Ramblas and Barceloneta. Use an inside pocket or money belt.",
                ],
                "best_time_to_visit": "May-June and September-October offer warm weather without the peak summer heat and crowds.",
            },
            "Rome": {
                "trip_name": "Roma Eterna: History, Art & La Dolce Vita",
                "overview": "Rome overwhelms and enchants in equal measure. A thoughtfully paced itinerary lets you absorb 3,000 years of history without museum fatigue, with evenings dedicated to the city's extraordinary food culture.",
                "days": [
                    {
                        "day": 1,
                        "theme": "Ancient Rome",
                        "morning": "Start at the Colosseum at opening time (book online). Combined ticket includes the Roman Forum and Palatine Hill — allow a full morning.",
                        "afternoon": "Walk to the Circus Maximus, then up to the Aventine Hill for the Knights of Malta keyhole view of St. Peter's dome.",
                        "evening": "Dinner in Testaccio — try Da Remo for the city's best pizza al taglio, or Grazia & Graziella for classic Roman trattoria cooking.",
                        "transport_tip": "The Colosseum, Forum, and Palatine are walkable. Metro Line B at Colosseo connects to Termini.",
                        "estimated_daily_budget": "€80-120 per person",
                    },
                    {
                        "day": 2,
                        "theme": "Vatican & Trastevere",
                        "morning": "Vatican Museums and Sistine Chapel (pre-booked skip-the-line mandatory). Allow 3.5-4 hours. St. Peter's Basilica is free — climb the dome.",
                        "afternoon": "Cross the Tiber to Trastevere for a long, leisurely Roman lunch. Supplì Roma for fried rice balls first.",
                        "evening": "Trastevere comes alive after 7pm. Dinner at Da Enzo al 29 for definitive cacio e pepe (reserve ahead).",
                        "transport_tip": "Tram 8 from Largo Argentina to Trastevere. Walk back across Ponte Sisto for sunset views.",
                        "estimated_daily_budget": "€90-130 per person",
                    },
                    {
                        "day": 3,
                        "theme": "Baroque Rome & Hidden Gems",
                        "morning": "Visit the Pantheon (now ticketed, arrive early). Walk to the Trevi Fountain at 7am before the crowds — magical in early light.",
                        "afternoon": "Borghese Gallery (strict advance booking, max 360 visitors). The Bernini sculptures alone justify the visit.",
                        "evening": "Aperitivo on Piazza Navona, then dinner at Roscioli — extraordinary wine cellar, exceptional Roman cuisine.",
                        "transport_tip": "The Centro Storico is best explored on foot. The Borghese is a 20-minute walk from Piazza del Popolo.",
                        "estimated_daily_budget": "€100-150 per person",
                    },
                ],
                "total_estimated_cost": "€270-400 per person (excluding flights & hotel)",
                "practical_tips": [
                    "Book everything in advance: Vatican, Borghese, Colosseum. Rome's top sights consistently sell out.",
                    "Carry a reusable water bottle — Rome's nasoni street fountains provide free, cold, clean water city-wide.",
                    "The cover charge (coperto, €2-4) at restaurants is standard and not a scam. Service is not included — round up for good service.",
                ],
                "best_time_to_visit": "April-May and October-November for mild weather, fewer tourists, and lower prices.",
            },
            "Lisbon": {
                "trip_name": "Lisboa: Fado, Pastéis & the Seven Hills",
                "overview": "Lisbon is Europe's most underrated capital — compact, characterful, and extraordinarily warm. This itinerary balances iconic sights with neighbourhood discoveries and the city's magnificent food and music culture.",
                "days": [
                    {
                        "day": 1,
                        "theme": "Alfama & Fado",
                        "morning": "Ride Tram 28 through Alfama to the Miradouro da Graça for morning coffee with city views. Walk down through the ancient Moorish quarter.",
                        "afternoon": "São Jorge Castle for panoramic Tagus views, then the Museu do Fado to understand Portugal's soulful music tradition.",
                        "evening": "Dinner and live fado at Clube de Fado in Alfama — book well ahead. The combination of food and authentic fadistas is extraordinary.",
                        "transport_tip": "Tram 28 requires patience (queues) but is a quintessential Lisbon experience. Buy the Viva Viagem card for transport.",
                        "estimated_daily_budget": "€70-100 per person",
                    },
                    {
                        "day": 2,
                        "theme": "Belém & Riverside",
                        "morning": "Train from Cais do Sodré to Belém (15 min). Jerónimos Monastery first, then Belém Tower, then pastéis de nata at Pastéis de Belém (the original).",
                        "afternoon": "MAAT Museum on the riverfront for contemporary art with a superb Tagus view from the roof. Then walk the riverside promenade.",
                        "evening": "Time Out Market for dinner — a culinary tour of Portugal under one roof. Don't miss the bacalhau and the bifana.",
                        "transport_tip": "Line 15E tram from Praça da Figueira runs directly to Belém. Journey time: 25 minutes.",
                        "estimated_daily_budget": "€65-95 per person",
                    },
                    {
                        "day": 3,
                        "theme": "Sintra & Return",
                        "morning": "Early train from Rossio to Sintra (40 min, runs every 20 min). Pena Palace first — buy a combined ticket for the Moorish Castle.",
                        "afternoon": "Explore the Quinta da Regaleira gardens (mystical Initiation Well) and the historic town centre with its pastelaria.",
                        "evening": "Return to Lisbon and relax in Chiado. Dinner at Taberna da Rua das Flores — spontaneous menu based on the day's market finds.",
                        "transport_tip": "Buy a return train ticket to Sintra from Rossio station. Start early — Pena Palace queues build by 10am.",
                        "estimated_daily_budget": "€80-110 per person",
                    },
                ],
                "total_estimated_cost": "€215-305 per person (excluding flights & hotel)",
                "practical_tips": [
                    "The Lisboa Card (24/48/72h) covers all public transport plus free entry to most museums. Excellent value.",
                    "Lisbon is hilly — comfortable walking shoes are essential. The historic trams are slow but charming.",
                    "Visit Sintra on a weekday and arrive before 9am to experience the palaces before the tour groups arrive.",
                ],
                "best_time_to_visit": "March-May and October for mild weather, fewer crowds, and lower prices.",
            },
        }

        template = city_templates.get(city, city_templates["Amsterdam"])

        # Trim or extend days to match request
        day_templates = template["days"]
        while len(day_templates) < days:
            extra_day = {
                "day": len(day_templates) + 1,
                "theme": "Local Exploration",
                "morning": f"Explore a neighbourhood you haven't visited yet in {city}.",
                "afternoon": "Visit a local market or lesser-known museum.",
                "evening": "Try a restaurant recommended by your hotel concierge.",
                "transport_tip": "Ask your hotel for local neighbourhood recommendations.",
                "estimated_daily_budget": "€70-100 per person",
            }
            day_templates = day_templates + [extra_day]
        day_templates = day_templates[:days]

        result: dict[str, Any] = {
            "trip_name": template["trip_name"],
            "overview": template["overview"],
            "days": day_templates,
            "total_estimated_cost": template["total_estimated_cost"],
            "practical_tips": template["practical_tips"],
            "best_time_to_visit": template["best_time_to_visit"],
        }
        return json.dumps(result)

    def _mock_critique(self, user_content: str) -> str:
        return json.dumps(
            {
                "budget_respected": True,
                "duration_included": True,
                "activities_sufficient": True,
                "hotel_alignment": True,
                "assumptions_stated": True,
                "overall_score": 0.88,
                "flags": [],
                "suggestions": [
                    "Consider adding a specific local neighbourhood walk for a more authentic experience.",
                    "Mention that popular attractions require advance booking.",
                ],
                "approved": True,
            }
        )

    def _mock_final_answer(self, user_content: str) -> str:
        city_match = re.search(r"Destination: (\w+)", user_content)
        city = city_match.group(1) if city_match else "your destination"

        days_match = re.search(r"Duration: (\d+) days", user_content)
        days = days_match.group(1) if days_match else "a few"

        city_intros = {
            "Amsterdam": "Amsterdam is one of Europe's most rewarding city break destinations",
            "Paris": "Paris never disappoints, and your timing sounds perfect",
            "Barcelona": "Barcelona is one of Europe's most dynamic and sensory cities",
            "Rome": "Rome is the kind of city that stops you in your tracks",
            "Lisbon": "Lisbon is Europe's most warmly human capital city",
        }
        intro = city_intros.get(city, f"{city} is a wonderful destination")

        return f"""{intro}, and based on everything you've shared, I've put together a {days}-day experience I think you'll genuinely love.

I've recommended a handful of hotels that tick your key boxes — all are in excellent locations with strong guest reviews, and each one has been scored specifically against your stated preferences for location, transport access, and the kind of atmosphere you're looking for. The top pick stands out for its proximity to your main interests and its consistently outstanding guest scores.

For the itinerary itself, I've balanced your must-do highlights with some quieter, more local moments — because the best travel memories usually come from both. Each day has a clear theme, so you can move at a comfortable pace without feeling rushed. I've included specific restaurant and venue recommendations rather than generic suggestions, and flagged which spots need advance booking (several do, and it's worth sorting that before you go).

A few things worth noting: I've assumed you'll be happy walking between sights in the city centre, which is always the best way to discover a city's character. Where a day trip is included, I've picked the most accessible option that fits neatly into a single day. The estimated costs are per person and exclude flights and accommodation — just activities, food, and local transport.

My honest recommendation: book your first-choice hotel and the popular attraction tickets as soon as possible. The rest can be wonderfully spontaneous. Have a brilliant trip.
"""


# ---------------------------------------------------------------------------
# OpenAI-compatible provider
# ---------------------------------------------------------------------------
class OpenAICompatibleProvider(LLMProvider):
    """
    LLM provider for OpenAI and OpenAI-compatible APIs.

    Works with OpenAI, Azure OpenAI, Ollama (with OpenAI-compatible mode),
    and any other endpoint that follows the OpenAI chat completions spec.
    """

    def __init__(self) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError("Install 'openai' to use the OpenAI provider.") from exc

        settings = get_settings()
        self._client = OpenAI(
            api_key=settings.openai_api_key or "not-needed",
            base_url=settings.openai_base_url,
            timeout=settings.llm_timeout_seconds,
            max_retries=2,
        )
        self._model = settings.llm_model
        self._temperature = settings.llm_temperature
        self._max_tokens = settings.llm_max_tokens

    def generate(
        self,
        messages: list[Message],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        import time

        oai_messages = [{"role": m.role, "content": m.content} for m in messages]
        start = time.perf_counter()

        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=oai_messages,  # type: ignore[arg-type]
                temperature=temperature or self._temperature,
                max_tokens=max_tokens or self._max_tokens,
            )
        except Exception as exc:
            raise LLMError(f"OpenAI API call failed: {exc}") from exc

        elapsed_ms = (time.perf_counter() - start) * 1000
        usage = response.usage

        return LLMResponse(
            content=response.choices[0].message.content or "",
            model=response.model,
            provider="openai",
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            latency_ms=elapsed_ms,
            is_mock=False,
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def create_llm_provider(force_mock: bool = False) -> LLMProvider:
    """
    Instantiate the appropriate LLM provider based on configuration.

    Always falls back to MockLLMProvider if the real provider fails to
    initialise (e.g. missing API key in a demo environment).
    """
    settings = get_settings()

    if force_mock or settings.is_mock_llm:
        logger.info("llm.provider", provider="mock")
        return MockLLMProvider()

    try:
        provider = OpenAICompatibleProvider()
        logger.info("llm.provider", provider=settings.llm_provider, model=settings.llm_model)
        get_registry().counter("llm.real_provider_init").inc()
        return provider
    except Exception as exc:
        logger.warning(
            "llm.provider_fallback",
            reason=str(exc),
            fallback="mock",
        )
        get_registry().counter("llm.fallback_to_mock").inc()
        return MockLLMProvider()


def parse_json_response(response: LLMResponse, context: str = "") -> dict:
    """
    Extract and parse JSON from an LLM response.

    Handles the common case where the model wraps JSON in markdown code fences.
    """
    content = response.content.strip()

    # Strip markdown code fences if present
    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?\n?", "", content)
        content = re.sub(r"\n?```$", "", content)
        content = content.strip()

    try:
        return json.loads(content)
    except json.JSONDecodeError as exc:
        raise LLMParseError(
            raw_response=response.content, expected_format=f"JSON ({context})"
        ) from exc
