"""
SIH26071 – Risk Engine & Early Warning Generation Service
Integrates Heavy Rainfall Classification (Logistic Regression) + Flood Inundation Prediction (Decision Tree)
Computes compound disaster risk and generates calibrated early warning advisories.

Design Principle:
- Transparent, reproducible, and explainable risk rules (no black-box frontend calculations).
- Decoupled service layer.
"""

from typing import Dict, Any, Optional

class RiskEngine:
    """
    Evaluates meteorological hazard from Module A and hydrological inundation from Module B,
    combining them with hydraulic threshold monitoring to derive an overall disaster risk tier.
    """

    # Hazard classification thresholds
    THRESHOLD_LOW = 30.0
    THRESHOLD_MODERATE = 60.0
    THRESHOLD_HIGH = 80.0

    @classmethod
    def evaluate_rainfall_hazard(cls, probability: float, heavy_rainfall: bool, pred_mm: Optional[float] = None) -> Dict[str, Any]:
        """Categorizes heavy rainfall likelihood into standard early-warning tiers."""
        prob_pct = round(probability * 100, 1)

        if prob_pct < 25.0:
            level = 'LOW'
            status = 'NO'
            badge_color = '#10b981'
            msg = 'No heavy rainfall risk detected. Precipitation expected below 64.5 mm IMD threshold.'
            advisory = 'Routine agricultural and drainage conditions expected.'
        elif prob_pct < 50.0:
            level = 'MODERATE'
            status = 'LOW / ELEVATED'
            badge_color = '#f59e0b'
            msg = 'Moderate probability of elevated precipitation surge. Monitoring advised.'
            advisory = 'Localized showers possible. Check local drainage conduits.'
        elif prob_pct < 75.0:
            level = 'HIGH'
            status = 'YES'
            badge_color = '#f97316'
            msg = 'High probability of Heavy Rainfall exceeding IMD 64.5 mm threshold.'
            advisory = 'Water-logging and surface runoff likely. Alert local municipal authorities.'
        else:
            level = 'CRITICAL'
            status = 'YES'
            badge_color = '#ef4444'
            msg = 'CRITICAL: Severe precipitation storm event highly probable (>= 64.5 mm deluge).'
            advisory = 'High danger of flash flooding, riverbank overtopping, and transport disruption.'

        return {
            'heavy_rainfall': heavy_rainfall,
            'prediction_label': 'YES' if heavy_rainfall else 'NO',
            'probability': probability,
            'probability_pct': prob_pct,
            'risk_level': level,
            'badge_color': badge_color,
            'message': msg,
            'advisory': advisory,
            'threshold_criteria': 'IMD 24h Heavy Rainfall Standard (>= 64.5 mm)'
        }

    @classmethod
    def evaluate_flood_hazard(cls, probability: float, flood_occurred: bool, water_level_m: float = 0.0) -> Dict[str, Any]:
        """Categorizes inundation probability into operational warning tiers."""
        prob_pct = round(probability * 100, 1)

        if prob_pct < 35.0:
            level = 'LOW'
            badge_color = '#10b981'
            msg = 'Minimal inundation probability. River water level within safe limits.'
            evacuation = 'None'
        elif prob_pct < 60.0:
            level = 'MODERATE'
            badge_color = '#f59e0b'
            msg = 'Moderate flood risk. Catchment saturation and river discharge rising.'
            evacuation = 'Routine Vigilance'
        elif prob_pct < 80.0:
            level = 'HIGH'
            badge_color = '#f97316'
            msg = 'High flood risk alert. Hydraulic breach likely in low-lying zones.'
            evacuation = 'Precautionary Preparedness in Riparian Sectors'
        else:
            level = 'CRITICAL'
            badge_color = '#ef4444'
            msg = 'CRITICAL FLOOD HAZARD: High probability of widespread inundation.'
            evacuation = 'Immediate Evacuation to High Ground Recommended'

        return {
            'flood_risk': flood_occurred,
            'prediction_label': 'YES' if flood_occurred else 'NO',
            'probability': probability,
            'probability_pct': prob_pct,
            'risk_level': level,
            'badge_color': badge_color,
            'message': msg,
            'evacuation_advice': evacuation
        }

    @classmethod
    def compute_compound_risk(
        cls,
        rainfall_prob: float,
        flood_prob: float,
        environmental_stress: float = 0.0,
        location_meta: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Synthesizes Module A (Rainfall) and Module B (Flood) into an integrated early warning index.
        Formula:
          Compound Score = 0.45 * Rainfall_Prob + 0.45 * Flood_Prob + 0.10 * Hydraulic_Stress
        """
        # Normalized environmental stress between 0.0 and 1.0
        env_factor = min(max(environmental_stress, 0.0), 1.0)
        
        compound_score = (0.45 * rainfall_prob) + (0.45 * flood_prob) + (0.10 * env_factor)
        compound_pct = round(compound_score * 100, 1)

        if compound_pct < cls.THRESHOLD_LOW:
            tier = 'LOW'
            badge_color = '#10b981'
            bulletin = 'LOW RISK: Environmental parameters and meteorological forecasts indicate normal conditions.'
            recommendation = 'Maintain regular hydrological monitoring. No emergency defense required.'
        elif compound_pct < cls.THRESHOLD_MODERATE:
            tier = 'MODERATE'
            badge_color = '#f59e0b'
            bulletin = 'MODERATE ALERT: Elevated rainfall and catchment saturation detected. Monitor vulnerable drainage sectors.'
            recommendation = 'Alert local maintenance teams; inspect sluice gates and storm drains.'
        elif compound_pct < cls.THRESHOLD_HIGH:
            tier = 'HIGH'
            badge_color = '#f97316'
            bulletin = 'HIGH RISK WARNING: Coincident heavy rainfall and flood inundation probability detected!'
            recommendation = 'Mobilize local disaster preparedness units; prepare temporary shelter facilities.'
        else:
            tier = 'CRITICAL'
            badge_color = '#ef4444'
            bulletin = 'CRITICAL EMERGENCY: Severe heavy rainfall probability combined with high flood susceptibility!'
            recommendation = 'Immediate disaster response protocol: notify emergency personnel, issue community evacuation advisories.'

        return {
            'compound_risk_score': compound_pct,
            'overall_risk': tier,
            'badge_color': badge_color,
            'warning_bulletin': bulletin,
            'actionable_recommendations': recommendation,
            'weights_applied': {
                'rainfall_probability_weight': 0.45,
                'flood_probability_weight': 0.45,
                'hydraulic_stress_weight': 0.10
            },
            'disclaimer': 'AI/ML-based decision support prototype. Adhere to official NDMA/IMD bulletins.'
        }
