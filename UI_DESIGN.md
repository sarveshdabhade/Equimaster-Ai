# Research dashboard design

The dashboard keeps Equimaster's original dark background and cyan/green accents,
with a compact financial-research layout inspired by the public
[Trendlyne interface examples](https://help.trendlyne.com/support/solutions/articles/84000384324-how-do-i-get-started-with-the-trendlyne-app-).
Equimaster retains its own branding, data, calculations and model outputs.

## Application layout

- Shared navigation connects Markets and Stock analysis using native Streamlit
  page links. Model training and evaluation run in the backend pipeline, not in
  the frontend.
- Markets includes dated index quotes, benchmark history, coverage breadth, sector
  moves, gainers, losers and cached recent news. Breadth describes the companies
  covered by this project, rather than the entire exchange.
- Stock screener filters the saved coverage by sector, company name, P/E, ROE,
  debt/equity or recent returns. Its company links open the relevant research URL;
  the filtered results can be exported. Generic leverage screens exclude financials.
- Stock analysis includes a company quote, financial snapshot, valuation context
  and price momentum. The six research sections render on selection, avoiding
  unnecessary model imports and news retrieval on the default Overview page.
- Overview provides price periods, line/candle charts, key financial facts,
  a 52-week range, historical returns and CSV export. Existing technical charts,
  forecasting, sentiment, fundamentals, statements and model management remain available.

Quotes and saved history show their dates separately. Missing data remains
unavailable. Research cards use actual ratios and returns; no proprietary scores
or synthetic market movements are substituted.

## Streamlit implementation

`.streamlit/config.toml` defines the native dark theme, including dropdowns, alerts
and dataframe headers. `assets/app.css` controls
spacing, colors, responsive panels and navigation. `components/layout.py` loads
the stylesheet, shared chrome and Plotly template. Start the app from `app.py` so
the multipage registry is initialized. Restart Streamlit after editing its theme
configuration or imported UI modules.

Most controls remain native Streamlit widgets. CSS uses stable container keys and
Streamlit test attributes; check these selectors when upgrading Streamlit. The
research navigation is a native keyboard-accessible radio group styled as tabs.
Small screens use a two-row header and horizontally scrolling index/research strips.
Tables scroll within their own containers.

Keep foreground and background colors paired when editing the theme. Small labels
use a readable muted color at full opacity; selected multiselect chips use light
text on a dark cyan background. All Plotly sections use the shared `CHART_STYLE`
and dark `equimaster` template. Fundamentals assessment cells define both dark
backgrounds and matching light text. The stylesheet loader accepts UTF-8 with or
without a BOM so Windows editors cannot invalidate its leading palette rule.

## Verification

Run the offline regression and interaction suite:

```powershell
python -m pytest -q
```

For optional screenshots, install the browser tooling separately. The helper uses
the existing Windows Edge installation; it does not download a browser:

```powershell
python -m pip install -r requirements-ui.txt
python -m streamlit run app.py --server.port 8511
# In another terminal:
python scripts/ui_screenshot.py --url http://127.0.0.1:8511 --output .cache/ui_review/markets.png --wait-for "Coverage breadth"
python scripts/ui_screenshot.py --url "http://127.0.0.1:8511/stock_analyzer?ticker=TCS.NS" --output .cache/ui_review/stock_mobile.png --width 390 --height 844 --wait-for "Key facts"
```

Generated screenshots are ignored under `.cache/ui_review/`.

The release acceptance runner is `python scripts/validate_browser.py` against an
app on port 8502. It checks exports, forecast generation, research sections,
unavailable-symbol/model states and a 390-pixel layout. Release evidence is saved
to `data/evaluation/browser_validation.json`. The sidebar uses Streamlit's automatic
initial state so a fresh mobile page starts with the navigation collapsed.
