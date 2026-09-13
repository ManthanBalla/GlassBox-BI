"""Generic Business Data Contract and Dataset-Specific Column Mapping.

Establishes a universal, organization-agnostic schema representation for business
time-series records. Enables arbitrary tabular/CSV datasets (synthetic, Walmart,
Rossmann, etc.) to map their source columns into a canonical contract.
"""

import datetime as dt
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field, field_validator


class BusinessTimeSeriesRecord(BaseModel):
    """Canonical, organization-agnostic business time-series record contract."""
    # Required core attributes for supervised time-series forecasting
    date: Union[str, dt.date, dt.datetime] = Field(
        ...,
        description="Observation date/timestamp for the time-series record",
    )
    entity_id: str = Field(
        ...,
        min_length=1,
        description="Unique business entity identifier (e.g., STORE_001, REGION_A, ORG_10)",
    )
    target: float = Field(
        ...,
        description="Primary historical metric/demand to be forecast (e.g., sales quantity, revenue)",
    )

    # Optional standard business features
    product_id: Optional[str] = Field(
        default=None,
        description="Product or SKU identifier",
    )
    category: Optional[str] = Field(
        default=None,
        description="Product or business category",
    )
    region: Optional[str] = Field(
        default=None,
        description="Geographic region or territory",
    )
    store_type: Optional[str] = Field(
        default=None,
        description="Store format or business classification",
    )
    price: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Unit selling price",
    )
    promotion: Optional[int] = Field(
        default=None,
        description="Promotion indicator (binary 0/1 or categorical promo flag)",
    )
    holiday: Optional[int] = Field(
        default=None,
        description="Holiday or special event indicator (binary 0/1 or code)",
    )
    inventory: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Available on-hand inventory or stock level",
    )

    # Dynamic extension dictionary for non-standard or custom regressors
    additional_features: Dict[str, Any] = Field(
        default_factory=dict,
        description="Unmapped or custom exogenous features preserved from source data",
    )

    @field_validator("date", mode="before")
    @classmethod
    def parse_date_string(cls, v: Any) -> Any:
        if isinstance(v, str):
            # Parse ISO or standard YYYY-MM-DD
            try:
                return dt.datetime.fromisoformat(v.replace("Z", "+00:00")).date()
            except ValueError:
                try:
                    return dt.datetime.strptime(v, "%Y-%m-%d").date()
                except ValueError:
                    return v
        return v

    @field_validator("target")
    @classmethod
    def validate_target_non_negative(cls, v: float) -> float:
        if v < 0:
            raise ValueError(f"Target value cannot be negative in retail demand contexts, got {v}")
        return v


class ColumnMapping(BaseModel):
    """Specification mapping dataset-specific source columns to canonical contract fields."""
    date: str = Field(default="date", description="Source column name for observation date")
    entity_id: str = Field(default="entity_id", description="Source column name for entity ID")
    target: str = Field(default="target", description="Source column name for target metric")
    product_id: Optional[str] = Field(default="product_id", description="Source column for product ID")
    category: Optional[str] = Field(default="category", description="Source column for category")
    region: Optional[str] = Field(default="region", description="Source column for region")
    store_type: Optional[str] = Field(default="store_type", description="Source column for store type")
    price: Optional[str] = Field(default="price", description="Source column for price")
    promotion: Optional[str] = Field(default="promotion", description="Source column for promotion flag")
    holiday: Optional[str] = Field(default="holiday", description="Source column for holiday flag")
    inventory: Optional[str] = Field(default="inventory", description="Source column for inventory")
    custom_mappings: Dict[str, str] = Field(
        default_factory=dict,
        description="Optional dictionary mapping additional source columns to custom feature names",
    )

    def get_source_column(self, canonical_field: str) -> Optional[str]:
        """Returns the source column name corresponding to a canonical field."""
        return getattr(self, canonical_field, None) or self.custom_mappings.get(canonical_field)


# ==============================================================================
# Pre-configured Source Adapters (Reusable Templates for Real Benchmarks)
# ==============================================================================

DEFAULT_CANONICAL_MAPPING = ColumnMapping(
    date="date",
    entity_id="entity_id",
    target="target",
    product_id="product_id",
    category="category",
    region="region",
    store_type="store_type",
    price="price",
    promotion="promotion",
    holiday="holiday",
    inventory="inventory",
)

SYNTHETIC_RETAIL_MAPPING = ColumnMapping(
    date="date",
    entity_id="entity_id",
    target="target",
    product_id="product_id",
    category="category",
    region="region",
    store_type="store_type",
    price="price",
    promotion="promotion",
    holiday="holiday",
    inventory="inventory",
)

# Walmart Recruiting - Store Sales Forecasting benchmark template
WALMART_MAPPING_TEMPLATE = ColumnMapping(
    date="Date",
    entity_id="Store",
    target="Weekly_Sales",
    product_id="Dept",
    holiday="IsHoliday",
    category=None,
    region=None,
    store_type=None,
    price=None,
    promotion=None,
    inventory=None,
    custom_mappings={
        "Temperature": "temperature",
        "Fuel_Price": "fuel_price",
        "CPI": "cpi",
        "Unemployment": "unemployment",
    },
)

# Rossmann Store Sales benchmark template
ROSSMANN_MAPPING_TEMPLATE = ColumnMapping(
    date="Date",
    entity_id="Store",
    target="Sales",
    promotion="Promo",
    holiday="StateHoliday",
    product_id=None,
    category=None,
    region=None,
    store_type=None,
    price=None,
    inventory=None,
    custom_mappings={
        "Customers": "customers",
        "SchoolHoliday": "school_holiday",
    },
)


def auto_detect_column_mapping(columns: List[str]) -> ColumnMapping:
    """Heuristically infers column mapping by inspecting header names for standard synonyms."""
    col_lower = {c.lower(): c for c in columns}

    def _match(synonyms: List[str], default: Optional[str] = None) -> Optional[str]:
        # 1. Exact match
        for s in synonyms:
            if s in col_lower:
                return col_lower[s]
        # 2. Substring match
        for col_key, original_col in col_lower.items():
            for s in synonyms:
                if s in col_key:
                    return original_col
        return default

    # Date synonyms
    date_col = _match(["date", "ds", "timestamp", "time", "datetime", "trans_date", "day"], default="date") or "date"

    # Target synonyms
    target_col = _match(["target", "weekly_sales", "sales", "demand", "quantity", "qty", "revenue", "units_sold", "y"], default="target") or "target"

    # Entity synonyms
    entity_col = _match(["entity_id", "store", "store_id", "shop_id", "location_id", "series_id", "customer_id"], default="entity_id") or "entity_id"

    # Product synonyms
    product_col = _match(["product_id", "dept", "item_id", "item_sku", "sku", "product", "item"])

    # Category synonyms
    category_col = _match(["category", "dept_name", "product_category", "group", "family"])

    # Price synonyms
    price_col = _match(["price", "unit_price", "selling_price", "mrp"])

    # Promotion synonyms
    promotion_col = _match(["promotion", "promo", "is_promo", "discount", "on_promotion"])

    # Holiday synonyms
    holiday_col = _match(["holiday", "isholiday", "stateholiday", "schoolholiday", "is_holiday"])

    # Inventory synonyms
    inventory_col = _match(["inventory", "stock", "stock_level", "on_hand"])

    # Region synonyms
    region_col = _match(["region", "state", "city", "country", "zone"])

    # Store type synonyms
    store_type_col = _match(["store_type", "store_format", "format", "type"])

    return ColumnMapping(
        date=date_col,
        entity_id=entity_col,
        target=target_col,
        product_id=product_col,
        category=category_col,
        region=region_col,
        store_type=store_type_col,
        price=price_col,
        promotion=promotion_col,
        holiday=holiday_col,
        inventory=inventory_col,
    )
