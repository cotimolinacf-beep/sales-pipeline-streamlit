"""
Sales Pipeline Analyzer - Streamlit UI
Step-by-step wizard flow.
"""

import streamlit as st
import json
import pandas as pd
from dotenv import load_dotenv

from models import (
    Pipeline,
    Objective,
    FieldDistribution,
    EvaluationResults,
    PIPELINE_TYPES,
    get_stages_for_type,
)
from data_processor import (
    load_csv,
    validate_sample_diversity,
    get_sample_for_auto_detect,
    conversations_to_prompt_text,
)
from graph import SalesPipelineAnalyzer
from charts import (
    funnel_chart,
    pipeline_funnel_chart,
    pipeline_goals_chart,
    keyword_distribution_chart,
    abandonment_chart,
)

load_dotenv()

st.set_page_config(
    page_title="Sales Pipeline Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# Session state defaults
# ──────────────────────────────────────────────

DEFAULTS = {
    "current_step": 1,
    "pipelines": [],
    "evaluation_results": None,
    "uploaded_df": None,
    "auto_detect_results": None,
}
for key, val in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = val


# ──────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────

with st.sidebar:
    st.header("📊 Sales Pipeline Analyzer")
    st.caption("Evaluacion de experiencia conversacional")

    st.divider()

    # LLM Config
    st.subheader("Proveedor LLM")
    provider = st.selectbox(
        "Proveedor",
        ["Google Gemini", "OpenAI", "Anthropic"],
        index=0,
        label_visibility="collapsed",
    )
    api_key = st.text_input("API Key", type="password")

    st.divider()

    # Pipeline import/export
    st.subheader("Configuracion")
    uploaded_config = st.file_uploader(
        "Importar pipelines (JSON)", type=["json"], key="config_upload"
    )
    if uploaded_config:
        try:
            config_data = json.loads(uploaded_config.read().decode("utf-8"))
            loaded = [Pipeline.from_dict(p) for p in config_data.get("pipelines", [])]
            st.session_state.pipelines = loaded
            if st.session_state.current_step < 2:
                st.session_state.current_step = 2
            st.success(f"{len(loaded)} pipeline(s) importados")
        except Exception as e:
            st.error(f"Error: {e}")

    if st.session_state.pipelines:
        config_json = json.dumps(
            {"pipelines": [p.to_dict() for p in st.session_state.pipelines]},
            ensure_ascii=False,
            indent=2,
        )
        st.download_button("Exportar pipelines JSON", config_json, "config.json", "application/json")

    st.divider()

    # Navigation
    st.subheader("Progreso")
    steps = {
        1: "Cargar CSV",
        2: "Configurar Pipeline",
        3: "Ejecutar Analisis",
        4: "Reporte",
    }
    for num, label in steps.items():
        if num < st.session_state.current_step:
            st.markdown(f"~~{num}. {label}~~  ✅")
        elif num == st.session_state.current_step:
            st.markdown(f"**{num}. {label}** ◀")
        else:
            st.markdown(f"{num}. {label}")

    st.divider()
    if st.button("Reiniciar todo", use_container_width=True):
        for key, val in DEFAULTS.items():
            st.session_state[key] = val
        st.rerun()


# ──────────────────────────────────────────────
# Step indicator
# ──────────────────────────────────────────────

step = st.session_state.current_step
step_labels = ["Cargar CSV", "Configurar Pipeline", "Ejecutar Analisis", "Reporte"]

cols_step = st.columns(4)
for i, label in enumerate(step_labels):
    num = i + 1
    with cols_step[i]:
        if num < step:
            st.success(f"**{num}.** {label} ✅", icon="✅")
        elif num == step:
            st.info(f"**{num}.** {label}", icon="👉")
        else:
            st.container(border=True).markdown(
                f"<div style='text-align:center;color:#94a3b8;padding:8px'>"
                f"<b>{num}.</b> {label}</div>",
                unsafe_allow_html=True,
            )

st.divider()

# ══════════════════════════════════════════════
# INTRO (shown only on step 1 before CSV upload)
# ══════════════════════════════════════════════

if step == 1:
    st.markdown(
        """
        <div style="
            background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
            border-radius: 12px;
            padding: 2rem 2.5rem;
            margin-bottom: 1.5rem;
            color: white;
        ">
            <h2 style="margin:0 0 0.5rem 0; color:white;">Bienvenido al Sales Pipeline Analyzer</h2>
            <p style="margin:0 0 1rem 0; font-size:1.1rem; opacity:0.9;">
                Evalua la experiencia conversacional de tus bots y agentes de atencion al cliente
                usando inteligencia artificial.
            </p>
            <div style="display:flex; gap:2rem; flex-wrap:wrap;">
                <div>
                    <span style="font-size:1.5rem;">📂</span><br>
                    <b>1. Carga tu CSV</b><br>
                    <span style="opacity:0.8; font-size:0.9rem;">Conversaciones con historial y tipificaciones</span>
                </div>
                <div>
                    <span style="font-size:1.5rem;">⚙️</span><br>
                    <b>2. Configura pipelines</b><br>
                    <span style="opacity:0.8; font-size:0.9rem;">Auto-deteccion con IA o configuracion manual</span>
                </div>
                <div>
                    <span style="font-size:1.5rem;">🤖</span><br>
                    <b>3. Analisis con IA</b><br>
                    <span style="opacity:0.8; font-size:0.9rem;">Funnel, sentimiento, friccion y abandono</span>
                </div>
                <div>
                    <span style="font-size:1.5rem;">📊</span><br>
                    <b>4. Reporte completo</b><br>
                    <span style="opacity:0.8; font-size:0.9rem;">KPIs, graficos y sugerencias accionables</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════
# STEP 1: CARGAR CSV
# ══════════════════════════════════════════════

    st.header("1. Cargar conversaciones")
    st.markdown("Sube el CSV con las conversaciones a analizar. "
                "Columnas requeridas: **historial** y **tipificaciones**. "
                "La columna **etapas** es opcional.")

    csv_file = st.file_uploader(
        "Selecciona tu archivo CSV",
        type=["csv"],
        key="main_csv",
    )

    if csv_file:
        try:
            df = load_csv(csv_file)
            st.session_state.uploaded_df = df

            col1, col2, col3 = st.columns(3)
            col1.metric("Conversaciones", f"{len(df):,}")
            col2.metric("Columnas detectadas", len(df.columns))

            diversity = validate_sample_diversity(df)
            col3.metric("Tipificaciones unicas", diversity["tipificaciones_count"])

            with st.expander("Vista previa de datos", expanded=True):
                st.dataframe(df.head(8), use_container_width=True)

            # Show detected columns
            st.caption(f"Columnas en el CSV: {', '.join(df.columns.tolist())}")

            st.divider()
            if st.button("Siguiente →", type="primary", use_container_width=True):
                st.session_state.current_step = 2
                st.rerun()

        except ValueError as e:
            st.error(str(e))
    else:
        st.info("Arrastra o selecciona un archivo CSV para comenzar.")


# ══════════════════════════════════════════════
# STEP 2: CONFIGURAR PIPELINE
# ══════════════════════════════════════════════

elif step == 2:
    st.header("2. Configurar Pipelines")

    config_mode = st.radio(
        "Como quieres configurar los pipelines?",
        ["Auto-detectar desde conversaciones", "Configuracion manual"],
        horizontal=True,
    )

    # ── AUTO-DETECT ──
    if config_mode == "Auto-detectar desde conversaciones":
        if st.session_state.uploaded_df is None:
            st.warning("Primero carga un CSV en el paso anterior.")
        else:
            df = st.session_state.uploaded_df
            st.markdown(f"Se usara una muestra de hasta **20 conversaciones** "
                        f"de las {len(df):,} cargadas para identificar pipelines.")

            if not st.session_state.auto_detect_results:
                if st.button("Detectar pipelines con IA", type="primary", use_container_width=True):
                    if not api_key:
                        st.error("Configura tu API Key en el sidebar.")
                    else:
                        with st.spinner("Analizando conversaciones con IA..."):
                            analyzer = SalesPipelineAnalyzer(provider, api_key)
                            sample = get_sample_for_auto_detect(df)
                            sample_text = conversations_to_prompt_text(sample)
                            suggested = analyzer.auto_detect_pipelines(sample_text)

                        if suggested:
                            st.session_state.auto_detect_results = suggested
                            st.rerun()
                        else:
                            st.error("No se pudieron detectar pipelines.")

        # Editable suggested pipelines
        if st.session_state.auto_detect_results:
            st.subheader("Pipelines detectados")
            st.caption("Edita nombre, objetivos, etapas y keywords antes de confirmar.")

            pipelines_to_remove = []
            for i, p_data in enumerate(st.session_state.auto_detect_results):
                p_type = p_data.get("type", "ventas")
                objectives = p_data.get("objectives", [])
                type_icon = "🛒" if p_type == "ventas" else "🎧"

                with st.container(border=True):
                    # Header
                    col_h1, col_h2, col_h3 = st.columns([3, 1.5, 0.5])
                    with col_h1:
                        p_data["name"] = st.text_input(
                            "Pipeline", value=p_data.get("name", ""),
                            key=f"ad_pn_{i}",
                        )
                    with col_h2:
                        new_type = st.selectbox(
                            "Tipo", PIPELINE_TYPES,
                            index=PIPELINE_TYPES.index(p_type) if p_type in PIPELINE_TYPES else 0,
                            format_func=lambda x: "🛒 Ventas" if x == "ventas" else "🎧 Servicio",
                            key=f"ad_pt_{i}",
                        )
                        p_data["type"] = new_type
                    with col_h3:
                        st.write("")
                        if st.button("🗑", key=f"ad_dp_{i}", help="Eliminar pipeline"):
                            pipelines_to_remove.append(i)

                    p_data["description"] = st.text_input(
                        "Descripcion", value=p_data.get("description", ""),
                        key=f"ad_pd_{i}",
                    )

                    # Objectives
                    if objectives:
                        objs_to_remove = []
                        stages = get_stages_for_type(new_type)

                        for j, obj in enumerate(objectives):
                            with st.container(border=True):
                                c1, c2, c3, c4 = st.columns([2.5, 1.5, 1, 0.4])
                                with c1:
                                    obj["name"] = st.text_input(
                                        "Objetivo", value=obj.get("name", ""),
                                        key=f"ad_on_{i}_{j}",
                                    )
                                with c2:
                                    cur = obj.get("stage", stages[0])
                                    idx = stages.index(cur) if cur in stages else 0
                                    obj["stage"] = st.selectbox(
                                        "Etapa", stages, index=idx, key=f"ad_os_{i}_{j}",
                                    )
                                with c3:
                                    obj["isConversionIndicator"] = st.checkbox(
                                        "⭐ Conversion", value=obj.get("isConversionIndicator", False),
                                        key=f"ad_oc_{i}_{j}",
                                    )
                                with c4:
                                    st.write("")
                                    if st.button("✕", key=f"ad_do_{i}_{j}"):
                                        objs_to_remove.append(j)

                                sc1, sc2 = st.columns(2)
                                with sc1:
                                    obj["success"] = st.text_input(
                                        "Criterio exito", value=obj.get("success", ""),
                                        key=f"ad_osu_{i}_{j}",
                                    )
                                with sc2:
                                    obj["failure"] = st.text_input(
                                        "Criterio fallo", value=obj.get("failure", ""),
                                        key=f"ad_of_{i}_{j}",
                                    )

                                fd_list = obj.get("field_distribution", [])
                                kw_str = ", ".join(kw for fd in fd_list for kw in fd.get("keywords", []))
                                new_kw = st.text_input(
                                    "Keywords (coma separados)", value=kw_str,
                                    key=f"ad_ok_{i}_{j}",
                                )
                                if new_kw.strip():
                                    kws = [k.strip() for k in new_kw.split(",") if k.strip()]
                                    obj["field_distribution"] = [{"name": "keywords", "keywords": kws}]
                                else:
                                    obj["field_distribution"] = []

                        for j in sorted(objs_to_remove, reverse=True):
                            objectives.pop(j)
                            st.rerun()

            for i in sorted(pipelines_to_remove, reverse=True):
                st.session_state.auto_detect_results.pop(i)
                st.rerun()

            # Actions
            st.divider()
            c_btn1, c_btn2, c_btn3 = st.columns([2, 1, 1])
            with c_btn1:
                if st.button("Confirmar pipelines →", type="primary", use_container_width=True):
                    loaded = [Pipeline.from_dict(p) for p in st.session_state.auto_detect_results]
                    st.session_state.pipelines = loaded
                    st.session_state.auto_detect_results = None
                    st.session_state.current_step = 3
                    st.rerun()
            with c_btn2:
                if st.button("Descartar", use_container_width=True):
                    st.session_state.auto_detect_results = None
                    st.rerun()
            with c_btn3:
                if st.button("← Volver", use_container_width=True):
                    st.session_state.current_step = 1
                    st.rerun()

    # ── MANUAL MODE ──
    else:
        # Existing pipelines
        if st.session_state.pipelines:
            for idx, pipeline in enumerate(st.session_state.pipelines):
                type_icon = "🛒" if pipeline.pipeline_type == "ventas" else "🎧"
                with st.expander(
                    f"{type_icon} {pipeline.name} — {len(pipeline.objectives)} objetivo(s)",
                    expanded=True,
                ):
                    col_p1, col_p2 = st.columns([4, 1])
                    with col_p1:
                        st.caption(f"{pipeline.pipeline_type.capitalize()} · {pipeline.description}")
                    with col_p2:
                        if st.button("Eliminar pipeline", key=f"dp_{idx}"):
                            st.session_state.pipelines.pop(idx)
                            st.rerun()

                    # Objectives table
                    for oi, obj in enumerate(pipeline.objectives):
                        star = "⭐ " if obj.is_conversion_indicator else ""
                        kw_text = ""
                        if obj.field_distribution:
                            kws = [kw for fd in obj.field_distribution for kw in fd.keywords]
                            if kws:
                                kw_text = f" · Keywords: {', '.join(kws[:5])}"

                        with st.container(border=True):
                            co1, co2, co3 = st.columns([3, 2, 0.5])
                            co1.markdown(f"**{star}{obj.name}**")
                            co2.caption(f"📍 {obj.stage}{kw_text}")
                            with co3:
                                if st.button("✕", key=f"do_{idx}_{oi}"):
                                    pipeline.objectives.pop(oi)
                                    if obj.is_conversion_indicator and pipeline.objectives:
                                        pipeline.objectives[0].is_conversion_indicator = True
                                    st.rerun()

                            cc1, cc2 = st.columns(2)
                            cc1.caption(f"✅ {obj.success}")
                            cc2.caption(f"❌ {obj.failure}")

                    # Add objective
                    st.markdown("---")
                    with st.form(key=f"add_obj_{idx}"):
                        st.markdown("**Agregar objetivo**")
                        stages = get_stages_for_type(pipeline.pipeline_type)
                        fc1, fc2 = st.columns(2)
                        obj_name = fc1.text_input("Nombre", max_chars=30, key=f"on_{idx}")
                        obj_stage = fc2.selectbox("Etapa", stages, key=f"os_{idx}")

                        fc3, fc4 = st.columns(2)
                        obj_success = fc3.text_input("Criterio exito", key=f"osu_{idx}")
                        obj_failure = fc4.text_input("Criterio fallo", key=f"of_{idx}")

                        fc5, fc6 = st.columns([3, 1])
                        obj_kw = fc5.text_input("Keywords (coma)", key=f"ok_{idx}")
                        obj_conv = fc6.checkbox(
                            "⭐ Conversion", key=f"oc_{idx}",
                            value=len(pipeline.objectives) == 0,
                        )

                        if st.form_submit_button("Agregar", use_container_width=True):
                            if obj_name and obj_success and obj_failure:
                                fd = []
                                if obj_kw.strip():
                                    kws = [k.strip() for k in obj_kw.split(",") if k.strip()]
                                    if kws:
                                        fd.append(FieldDistribution(name="keywords", keywords=kws))
                                new_obj = Objective(
                                    name=obj_name, stage=obj_stage,
                                    success=obj_success, failure=obj_failure,
                                    is_conversion_indicator=obj_conv, field_distribution=fd,
                                )
                                if obj_conv:
                                    for o in pipeline.objectives:
                                        o.is_conversion_indicator = False
                                pipeline.objectives.append(new_obj)
                                if len(pipeline.objectives) == 1:
                                    pipeline.objectives[0].is_conversion_indicator = True
                                st.rerun()
                            else:
                                st.warning("Completa nombre, criterio exito y criterio fallo.")
        else:
            st.info("Crea tu primer pipeline o importa un JSON desde el sidebar.")

        # New pipeline form
        st.divider()
        with st.form(key="new_pipeline"):
            st.markdown("**Nuevo pipeline**")
            np1, np2 = st.columns([3, 1])
            p_name = np1.text_input("Nombre", max_chars=30)
            p_type = np2.selectbox(
                "Tipo", PIPELINE_TYPES,
                format_func=lambda x: "🛒 Ventas" if x == "ventas" else "🎧 Servicio",
            )
            p_desc = st.text_input("Descripcion", max_chars=200)

            if st.form_submit_button("Crear pipeline", type="primary", use_container_width=True):
                if p_name:
                    existing = [p.name for p in st.session_state.pipelines]
                    if p_name in existing:
                        st.error("Ya existe un pipeline con ese nombre.")
                    else:
                        st.session_state.pipelines.append(
                            Pipeline(name=p_name, description=p_desc, pipeline_type=p_type)
                        )
                        st.rerun()
                else:
                    st.warning("Ingresa un nombre.")

        # Navigation
        st.divider()
        nav1, nav2 = st.columns(2)
        with nav1:
            if st.button("← Volver al CSV", use_container_width=True):
                st.session_state.current_step = 1
                st.rerun()
        with nav2:
            has_objectives = any(len(p.objectives) > 0 for p in st.session_state.pipelines)
            if st.session_state.pipelines and has_objectives:
                if st.button("Siguiente → Ejecutar analisis", type="primary", use_container_width=True):
                    st.session_state.current_step = 3
                    st.rerun()
            else:
                st.button(
                    "Siguiente → (agrega al menos 1 pipeline con objetivos)",
                    disabled=True, use_container_width=True,
                )


# ══════════════════════════════════════════════
# STEP 3: EJECUTAR ANALISIS
# ══════════════════════════════════════════════

elif step == 3:
    st.header("3. Ejecutar Analisis")

    df = st.session_state.uploaded_df
    pipelines = st.session_state.pipelines

    if df is None:
        st.warning("No hay CSV cargado.")
        if st.button("← Ir al paso 1"):
            st.session_state.current_step = 1
            st.rerun()
    elif not pipelines:
        st.warning("No hay pipelines configurados.")
        if st.button("← Ir al paso 2"):
            st.session_state.current_step = 2
            st.rerun()
    else:
        # Summary
        st.markdown("### Resumen del analisis")
        c1, c2, c3 = st.columns(3)
        c1.metric("Conversaciones", f"{len(df):,}")
        c2.metric("Pipelines", len(pipelines))
        total_obj = sum(len(p.objectives) for p in pipelines)
        c3.metric("Objetivos totales", total_obj)

        # Pipeline summary cards
        for p in pipelines:
            type_icon = "🛒" if p.pipeline_type == "ventas" else "🎧"
            conv_obj = p.get_conversion_indicator()
            conv_name = f" · ⭐ {conv_obj.name}" if conv_obj else ""
            with st.container(border=True):
                st.markdown(f"**{type_icon} {p.name}** — {len(p.objectives)} objetivo(s){conv_name}")
                stages_used = [o.stage for o in p.objectives]
                st.caption(f"Etapas: {' → '.join(stages_used)}")

        st.divider()

        # LLM status
        if not api_key:
            st.error("Configura tu API Key en el sidebar antes de ejecutar.")
        else:
            st.caption(f"Proveedor: **{provider}** · API Key configurada ✅")

            if st.button("Ejecutar analisis", type="primary", use_container_width=True, disabled=not api_key):
                progress_bar = st.progress(0)
                status_text = st.empty()

                def update_progress(value, text):
                    progress_bar.progress(value)
                    status_text.text(text)

                try:
                    analyzer = SalesPipelineAnalyzer(provider, api_key)
                    results = analyzer.run_analysis(df, pipelines, progress_callback=update_progress)
                    st.session_state.evaluation_results = results
                    progress_bar.progress(1.0)
                    status_text.text("Analisis completado!")
                    st.session_state.current_step = 4
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")
                    progress_bar.empty()
                    status_text.empty()

        # Navigation
        st.divider()
        if st.button("← Volver a pipelines", use_container_width=True):
            st.session_state.current_step = 2
            st.rerun()


# ══════════════════════════════════════════════
# STEP 4: REPORTE
# ══════════════════════════════════════════════

elif step == 4:
    results: EvaluationResults = st.session_state.evaluation_results

    if not results:
        st.warning("No hay resultados. Ejecuta el analisis primero.")
        if st.button("← Ir al paso 3"):
            st.session_state.current_step = 3
            st.rerun()
    else:
        st.header("4. Reporte de Analisis")

        # ── Global KPI bar ──
        with st.container(border=True):
            kpi_cols = st.columns(2 + len(results.pipelines))
            kpi_cols[0].metric("Total conversaciones", f"{results.total_conversations_analyzed:,}")
            if results.processing_time_seconds:
                kpi_cols[1].metric("Tiempo de analisis", f"{results.processing_time_seconds:.1f}s")

            for ki, pr in enumerate(results.pipelines):
                conv_goal = next((g for g in pr.funnel if g.is_conversion_indicator), None)
                if conv_goal:
                    rate = (conv_goal.success_count / pr.total_conversations * 100) if pr.total_conversations > 0 else 0
                    type_icon = "🛒" if pr.pipeline_type == "ventas" else "🎧"
                    kpi_cols[2 + ki].metric(
                        f"{type_icon} {pr.pipeline_name}",
                        f"{rate:.1f}%",
                        help=f"Conversion: {conv_goal.success_count}/{pr.total_conversations}",
                    )

        # ── CSV Download helper ──
        def build_export_df():
            df_source = st.session_state.uploaded_df
            if df_source is None:
                return pd.DataFrame()
            export = df_source.copy()
            conv_details = getattr(results, "conversation_details", []) or []
            if conv_details:
                # Add per-pipeline objective columns
                for pr in results.pipelines:
                    p_name = pr.pipeline_name
                    obj_cols: dict[str, list] = {}
                    kw_col = []
                    for i in range(len(export)):
                        if i < len(conv_details):
                            cd = conv_details[i]
                            pr_data = cd.pipeline_results.get(p_name, {})
                            objs = pr_data.get("objectives", {})
                            for obj_name, success in objs.items():
                                col_key = f"{p_name} | {obj_name}"
                                if col_key not in obj_cols:
                                    obj_cols[col_key] = [""] * i
                                obj_cols[col_key].append("SI" if success else "NO")
                            for col_key in obj_cols:
                                if len(obj_cols[col_key]) <= i:
                                    obj_cols[col_key].append("")
                            kws = pr_data.get("keywords", [])
                            kw_col.append(", ".join(kws) if kws else "")
                        else:
                            for col_key in obj_cols:
                                obj_cols[col_key].append("")
                            kw_col.append("")

                    for col_key, col_vals in obj_cols.items():
                        while len(col_vals) < len(export):
                            col_vals.append("")
                        export[col_key] = col_vals
                    export[f"{p_name} | keywords"] = kw_col

            return export

        # ── Pipeline tabs ──
        st.divider()

        tab_labels = []
        for pr in results.pipelines:
            type_icon = "🛒" if pr.pipeline_type == "ventas" else "🎧"
            tab_labels.append(f"{type_icon} {pr.pipeline_name}")
        tab_labels.append("📥 Descargar CSV")

        tabs = st.tabs(tab_labels)

        # ── Per-pipeline tabs (funnel + suggestions) ──
        category_icons = {
            "autogestion": "🤖",
            "conversion": "📈",
            "cuello_botella": "⚠️",
            "quick_win": "⚡",
        }
        category_labels_map = {
            "autogestion": "Autogestion",
            "conversion": "Conversion",
            "cuello_botella": "Cuellos de botella",
            "quick_win": "Quick Wins",
        }
        impact_colors = {"alto": "🔴", "medio": "🟡", "bajo": "🟢"}

        for ti, pr in enumerate(results.pipelines):
            with tabs[ti]:
                # Pipeline header metrics
                conv_goal = next((g for g in pr.funnel if g.is_conversion_indicator), None)
                conv_rate = conv_goal.success_rate if conv_goal else 0
                avg_rate = sum(g.success_rate for g in pr.funnel) / len(pr.funnel) if pr.funnel else 0

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Conversaciones", f"{pr.total_conversations:,}")
                m2.metric("Tasa de conversion", f"{conv_rate:.1f}%")
                m3.metric("Promedio exito", f"{avg_rate:.1f}%")
                if pr.abandonment_analysis:
                    m4.metric("Tasa abandono", f"{pr.abandonment_analysis.abandonment_rate:.1f}%")

                # Funnel chart
                st.plotly_chart(pipeline_funnel_chart(pr), use_container_width=True)

                # Goals detail
                st.markdown("#### Resultados por objetivo")
                st.plotly_chart(pipeline_goals_chart(pr), use_container_width=True)

                # Objective detail cards
                for goal in pr.funnel:
                    star = "⭐ " if goal.is_conversion_indicator else ""
                    with st.container(border=True):
                        gc1, gc2, gc3, gc4 = st.columns([3, 1, 1, 1])
                        gc1.markdown(f"**{star}{goal.objective_name}**")
                        gc1.caption(f"Etapa: {goal.stage}")
                        gc2.metric("Exitosos", goal.success_count)
                        gc3.metric("Fallidos", goal.failure_count)
                        gc4.metric("Tasa", f"{goal.success_rate:.1f}%")

                        if goal.keyword_distribution:
                            kw_tags = " · ".join(
                                f"`{kd.value}` ({kd.count})"
                                for kd in goal.keyword_distribution[:6]
                            )
                            st.caption(f"Keywords: {kw_tags}")

                # Abandonment reasons
                if pr.abandonment_analysis and pr.abandonment_analysis.top_abandonment_reasons:
                    st.markdown("#### Razones de abandono")
                    fig = abandonment_chart(pr)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)

                # ── Suggestions for this pipeline ──
                pr_suggestions = getattr(pr, "suggestions", []) or []
                if pr_suggestions:
                    st.divider()
                    st.markdown("#### 💡 Sugerencias de valor")

                    grouped: dict[str, list] = {}
                    for s in pr_suggestions:
                        grouped.setdefault(s.category, []).append(s)

                    for cat, items in grouped.items():
                        icon = category_icons.get(cat, "📋")
                        label = category_labels_map.get(cat, cat)
                        st.markdown(f"##### {icon} {label}")

                        for s in items:
                            badge = impact_colors.get(s.impact, "⚪")
                            with st.container(border=True):
                                sg1, sg2 = st.columns([5, 1])
                                sg1.markdown(f"**{s.title}**")
                                sg2.markdown(f"Impacto: {badge} {s.impact}")
                                st.markdown(s.description)
                                if s.metric:
                                    st.caption(f"📊 {s.metric}")

        # ── CSV Download tab ──
        csv_tab_idx = len(results.pipelines)
        with tabs[csv_tab_idx]:
            st.markdown("#### Descargar detalle del analisis")
            st.markdown(
                "El CSV incluye todas las conversaciones originales con columnas "
                "adicionales: resultado de **cada objetivo por pipeline** (SI/NO) "
                "y **keywords** encontrados."
            )

            export_df = build_export_df()
            if not export_df.empty:
                st.metric("Filas", f"{len(export_df):,}")
                st.caption(f"Columnas: {', '.join(export_df.columns.tolist())}")

                with st.expander("Vista previa", expanded=True):
                    st.dataframe(export_df.head(10), use_container_width=True)

                csv_data = export_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Descargar CSV con resultados",
                    csv_data,
                    "analisis_conversaciones.csv",
                    "text/csv",
                    type="primary",
                    use_container_width=True,
                )
            else:
                st.warning("No hay datos para exportar.")

        # ── Actions ──
        st.divider()
        act1, act2, act3 = st.columns(3)
        with act1:
            if st.button("← Volver al analisis", use_container_width=True):
                st.session_state.current_step = 3
                st.rerun()
        with act2:
            if st.button("Nuevo analisis (mismo pipeline)", use_container_width=True):
                st.session_state.evaluation_results = None
                st.session_state.uploaded_df = None
                st.session_state.current_step = 1
                st.rerun()
        with act3:
            if st.button("Reiniciar todo", use_container_width=True):
                for key, val in DEFAULTS.items():
                    st.session_state[key] = val
                st.rerun()
