"""
Streamlit UI for the Sales Pipeline Analyzer.
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
    SALES_STAGES,
    SERVICE_STAGES,
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
    pipeline_goals_chart,
    keyword_distribution_chart,
    sentiment_pie_chart,
    sentiment_bar_chart,
    abandonment_chart,
)

load_dotenv()

# ──────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────

st.set_page_config(
    page_title="Sales Pipeline Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# Session state defaults
# ──────────────────────────────────────────────

if "pipelines" not in st.session_state:
    st.session_state.pipelines = []
if "evaluation_results" not in st.session_state:
    st.session_state.evaluation_results = None
if "uploaded_df" not in st.session_state:
    st.session_state.uploaded_df = None
if "auto_detect_results" not in st.session_state:
    st.session_state.auto_detect_results = None


# ──────────────────────────────────────────────
# Sidebar: LLM Configuration
# ──────────────────────────────────────────────

with st.sidebar:
    st.header("Configuracion LLM")
    provider = st.selectbox(
        "Proveedor",
        ["Google Gemini", "OpenAI", "Anthropic"],
        index=0,
    )
    api_key = st.text_input("API Key", type="password")

    st.divider()

    # Pipeline config import/export
    st.subheader("Config de Pipelines")
    uploaded_config = st.file_uploader(
        "Cargar config JSON", type=["json"], key="config_upload"
    )
    if uploaded_config:
        try:
            config_data = json.loads(uploaded_config.read().decode("utf-8"))
            loaded_pipelines = [
                Pipeline.from_dict(p) for p in config_data.get("pipelines", [])
            ]
            st.session_state.pipelines = loaded_pipelines
            st.success(f"{len(loaded_pipelines)} pipeline(s) cargados")
        except Exception as e:
            st.error(f"Error al cargar config: {e}")

    if st.session_state.pipelines:
        config_json = json.dumps(
            {"pipelines": [p.to_dict() for p in st.session_state.pipelines]},
            ensure_ascii=False,
            indent=2,
        )
        st.download_button(
            "Descargar config JSON",
            config_json,
            "config.json",
            "application/json",
        )


# ──────────────────────────────────────────────
# Main content
# ──────────────────────────────────────────────

st.title("📊 Sales Pipeline Analyzer")
st.caption("Evaluacion de experiencia conversacional")

tab_config, tab_dashboard = st.tabs(["⚙️ Configuracion", "📈 Dashboard"])


# ══════════════════════════════════════════════
# TAB 1: CONFIGURACION
# ══════════════════════════════════════════════

with tab_config:
    config_mode = st.radio(
        "Modo de configuracion",
        ["Manual", "Auto-detectar desde CSV"],
        horizontal=True,
    )

    # ── Auto-detect mode ──
    if config_mode == "Auto-detectar desde CSV":
        st.subheader("Auto-detectar pipelines")
        st.info(
            "Sube una muestra de conversaciones y el sistema sugerira "
            "pipelines con objetivos, etapas y keywords automaticamente."
        )

        sample_file = st.file_uploader(
            "CSV de muestra", type=["csv"], key="sample_csv"
        )

        if sample_file:
            try:
                sample_df = load_csv(sample_file)
                diversity = validate_sample_diversity(sample_df)

                has_etapas = diversity["etapas_count"] > 0
                cols = st.columns(3 if has_etapas else 2)
                cols[0].metric("Conversaciones", diversity["total_conversations"])
                cols[1].metric("Tipificaciones unicas", diversity["tipificaciones_count"])
                if has_etapas:
                    cols[2].metric("Etapas unicas", diversity["etapas_count"])

                if not diversity["is_diverse"]:
                    st.warning(
                        "La muestra tiene poca diversidad. Intenta incluir conversaciones "
                        "de diferentes grupos/tipificaciones para mejores resultados."
                    )

                with st.expander("Preview de datos", expanded=False):
                    st.dataframe(sample_df.head(10), use_container_width=True)

                if st.button("Detectar pipelines", type="primary"):
                    if not api_key:
                        st.error("Configura tu API Key en el sidebar.")
                    else:
                        with st.spinner("Analizando conversaciones..."):
                            analyzer = SalesPipelineAnalyzer(provider, api_key)
                            sample = get_sample_for_auto_detect(sample_df)
                            sample_text = conversations_to_prompt_text(sample)
                            suggested = analyzer.auto_detect_pipelines(sample_text)

                        if suggested:
                            st.session_state.auto_detect_results = suggested
                            st.success(f"Se detectaron {len(suggested)} pipeline(s)")
                        else:
                            st.error("No se pudieron detectar pipelines. Intenta con una muestra mas diversa.")
            except ValueError as e:
                st.error(str(e))

        # Show auto-detected results for editing
        if st.session_state.auto_detect_results:
            st.subheader("Pipelines sugeridos (edita antes de confirmar)")
            for i, p_data in enumerate(st.session_state.auto_detect_results):
                with st.expander(f"Pipeline: {p_data.get('name', f'Pipeline {i+1}')}", expanded=True):
                    st.json(p_data)

            if st.button("Confirmar y cargar pipelines", type="primary"):
                loaded = [Pipeline.from_dict(p) for p in st.session_state.auto_detect_results]
                st.session_state.pipelines = loaded
                st.session_state.auto_detect_results = None
                st.success(f"{len(loaded)} pipeline(s) cargados a la configuracion")
                st.rerun()

    # ── Manual mode ──
    if config_mode == "Manual":
        st.subheader("Pipelines configurados")

        # Show existing pipelines
        if not st.session_state.pipelines:
            st.info("No hay pipelines configurados. Crea uno nuevo o carga un JSON desde el sidebar.")

        for idx, pipeline in enumerate(st.session_state.pipelines):
            with st.expander(
                f"{'🛒' if pipeline.pipeline_type == 'ventas' else '🎧'} "
                f"{pipeline.name} ({len(pipeline.objectives)} objetivos)",
                expanded=False,
            ):
                col_p1, col_p2 = st.columns([3, 1])
                with col_p1:
                    st.text(f"Tipo: {pipeline.pipeline_type} | {pipeline.description}")
                with col_p2:
                    if st.button("Eliminar", key=f"del_pipeline_{idx}"):
                        st.session_state.pipelines.pop(idx)
                        st.rerun()

                # Objectives list
                st.markdown("**Objetivos:**")
                for oi, obj in enumerate(pipeline.objectives):
                    indicator = " ⭐" if obj.is_conversion_indicator else ""
                    col_o1, col_o2, col_o3, col_o4 = st.columns([3, 1, 1, 1])
                    col_o1.text(f"{oi+1}. {obj.name}{indicator} [{obj.stage}]")
                    col_o2.text(f"✅ {obj.success[:30]}")
                    col_o3.text(f"❌ {obj.failure[:30]}")
                    if col_o4.button("🗑", key=f"del_obj_{idx}_{oi}"):
                        pipeline.objectives.pop(oi)
                        # Reassign conversion indicator if needed
                        if obj.is_conversion_indicator and pipeline.objectives:
                            pipeline.objectives[0].is_conversion_indicator = True
                        st.rerun()

                # Add objective form
                st.markdown("---")
                st.markdown("**Agregar objetivo:**")
                stages = get_stages_for_type(pipeline.pipeline_type)

                with st.form(key=f"add_obj_form_{idx}"):
                    obj_name = st.text_input("Nombre", max_chars=30, key=f"obj_name_{idx}")
                    obj_stage = st.selectbox("Etapa", stages, key=f"obj_stage_{idx}")
                    obj_success = st.text_input("Criterio de exito", key=f"obj_success_{idx}")
                    obj_failure = st.text_input("Criterio de fallo", key=f"obj_failure_{idx}")
                    obj_is_conv = st.checkbox(
                        "Indicador de conversion",
                        key=f"obj_conv_{idx}",
                        value=len(pipeline.objectives) == 0,
                    )
                    obj_keywords = st.text_input(
                        "Keywords (separados por coma)",
                        key=f"obj_kw_{idx}",
                        help="Ejemplo: Yaris, Tacoma, Hilux",
                    )

                    if st.form_submit_button("Agregar objetivo"):
                        if obj_name and obj_success and obj_failure:
                            # Parse keywords
                            fd_list = []
                            if obj_keywords.strip():
                                kws = [k.strip() for k in obj_keywords.split(",") if k.strip()]
                                if kws:
                                    fd_list.append(FieldDistribution(name="keywords", keywords=kws))

                            new_obj = Objective(
                                name=obj_name,
                                stage=obj_stage,
                                success=obj_success,
                                failure=obj_failure,
                                is_conversion_indicator=obj_is_conv,
                                field_distribution=fd_list,
                            )

                            # Enforce single conversion indicator
                            if obj_is_conv:
                                for o in pipeline.objectives:
                                    o.is_conversion_indicator = False

                            pipeline.objectives.append(new_obj)

                            # If first objective and not marked, mark it
                            if len(pipeline.objectives) == 1:
                                pipeline.objectives[0].is_conversion_indicator = True

                            st.rerun()
                        else:
                            st.warning("Completa todos los campos obligatorios.")

        # ── Create new pipeline ──
        st.divider()
        st.subheader("Nuevo pipeline")

        with st.form(key="new_pipeline_form"):
            p_name = st.text_input("Nombre del pipeline", max_chars=30)
            p_desc = st.text_area("Descripcion", max_chars=200)
            p_type = st.selectbox("Tipo", PIPELINE_TYPES, format_func=lambda x: "Ventas" if x == "ventas" else "Servicio")

            if st.form_submit_button("Crear pipeline", type="primary"):
                if p_name:
                    existing_names = [p.name for p in st.session_state.pipelines]
                    if p_name in existing_names:
                        st.error("Ya existe un pipeline con ese nombre.")
                    else:
                        new_pipeline = Pipeline(
                            name=p_name,
                            description=p_desc,
                            pipeline_type=p_type,
                        )
                        st.session_state.pipelines.append(new_pipeline)
                        st.rerun()
                else:
                    st.warning("Ingresa un nombre para el pipeline.")


# ══════════════════════════════════════════════
# TAB 2: DASHBOARD
# ══════════════════════════════════════════════

with tab_dashboard:
    if not st.session_state.pipelines:
        st.warning("Configura al menos un pipeline en la pestaña de Configuracion antes de analizar.")
    else:
        st.subheader("Cargar conversaciones")
        csv_file = st.file_uploader(
            "Sube tu CSV con conversaciones",
            type=["csv"],
            key="analysis_csv",
            help="Columnas requeridas: historial, tipificaciones, etapas",
        )

        if csv_file:
            try:
                df = load_csv(csv_file)
                st.session_state.uploaded_df = df

                col_u1, col_u2, col_u3 = st.columns(3)
                col_u1.metric("Conversaciones", len(df))
                col_u2.metric("Columnas", len(df.columns))
                col_u3.metric(
                    "Pipelines configurados", len(st.session_state.pipelines)
                )

                with st.expander("Preview de datos", expanded=False):
                    st.dataframe(df.head(5), use_container_width=True)

            except ValueError as e:
                st.error(str(e))
                st.session_state.uploaded_df = None

        # Run analysis button
        if st.session_state.uploaded_df is not None:
            if st.button("Ejecutar analisis", type="primary", use_container_width=True):
                if not api_key:
                    st.error("Configura tu API Key en el sidebar.")
                else:
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    def update_progress(value, text):
                        progress_bar.progress(value)
                        status_text.text(text)

                    try:
                        analyzer = SalesPipelineAnalyzer(provider, api_key)
                        results = analyzer.run_analysis(
                            st.session_state.uploaded_df,
                            st.session_state.pipelines,
                            progress_callback=update_progress,
                        )
                        st.session_state.evaluation_results = results
                        progress_bar.progress(1.0)
                        status_text.text("Analisis completado!")
                    except Exception as e:
                        st.error(f"Error durante el analisis: {e}")
                        progress_bar.empty()
                        status_text.empty()

        # ── Show results ──
        results: EvaluationResults = st.session_state.evaluation_results
        if results:
            st.divider()

            # ── KPI Cards ──
            st.subheader("Metricas principales")
            kpi_cols = st.columns(4)

            kpi_cols[0].metric(
                "Conversaciones analizadas",
                f"{results.total_conversations_analyzed:,}",
            )

            # Show conversion rate per pipeline
            kpi_idx = 1
            for pr in results.pipelines:
                conv_goal = next(
                    (g for g in pr.funnel if g.is_conversion_indicator), None
                )
                if conv_goal and kpi_idx < 4:
                    rate = (
                        (conv_goal.success_count / pr.total_conversations * 100)
                        if pr.total_conversations > 0
                        else 0
                    )
                    kpi_cols[kpi_idx].metric(
                        f"{pr.pipeline_name}",
                        f"{rate:.1f}%",
                        help=f"{conv_goal.objective_name}: {conv_goal.success_count}/{pr.total_conversations}",
                    )
                    kpi_idx += 1

            if results.processing_time_seconds:
                kpi_cols[min(kpi_idx, 3)].metric(
                    "Tiempo de procesamiento",
                    f"{results.processing_time_seconds:.1f}s",
                )

            # ── Global Funnel ──
            st.divider()
            st.plotly_chart(funnel_chart(results), use_container_width=True)

            # ── Pipeline Details ──
            st.divider()
            st.subheader("Detalle por pipeline")

            for pr in results.pipelines:
                with st.expander(
                    f"{'🛒' if pr.pipeline_type == 'ventas' else '🎧'} "
                    f"{pr.pipeline_name} ({pr.total_conversations} conversaciones)",
                    expanded=True,
                ):
                    # Summary metrics
                    col_pr1, col_pr2, col_pr3 = st.columns(3)
                    conv_goal = next(
                        (g for g in pr.funnel if g.is_conversion_indicator), None
                    )
                    conv_rate = conv_goal.success_rate if conv_goal else 0
                    avg_rate = (
                        sum(g.success_rate for g in pr.funnel) / len(pr.funnel)
                        if pr.funnel
                        else 0
                    )
                    col_pr1.metric("Tasa de conversion", f"{conv_rate:.1f}%")
                    col_pr2.metric("Promedio de exito", f"{avg_rate:.1f}%")
                    if pr.abandonment_analysis:
                        col_pr3.metric(
                            "Tasa de abandono",
                            f"{pr.abandonment_analysis.abandonment_rate:.1f}%",
                        )

                    # Goals bar chart
                    st.plotly_chart(
                        pipeline_goals_chart(pr), use_container_width=True
                    )

                    # Keyword distributions
                    for goal in pr.funnel:
                        kw_fig = keyword_distribution_chart(goal)
                        if kw_fig:
                            st.plotly_chart(kw_fig, use_container_width=True)

                    # Abandonment chart
                    ab_fig = abandonment_chart(pr)
                    if ab_fig:
                        st.plotly_chart(ab_fig, use_container_width=True)

            # ── Sentiment Analysis ──
            if results.sentiment_summary:
                st.divider()
                st.subheader("Analisis de sentimiento")

                ss = results.sentiment_summary
                col_s1, col_s2 = st.columns(2)

                with col_s1:
                    st.plotly_chart(
                        sentiment_pie_chart(ss.satisfied, ss.neutral, ss.frustrated),
                        use_container_width=True,
                    )

                with col_s2:
                    st.plotly_chart(
                        sentiment_bar_chart(ss.satisfied, ss.neutral, ss.frustrated),
                        use_container_width=True,
                    )

                    # Sentiment metrics
                    total = ss.satisfied + ss.neutral + ss.frustrated
                    if total > 0:
                        col_sm1, col_sm2, col_sm3 = st.columns(3)
                        col_sm1.metric(
                            "😊 Satisfechos",
                            ss.satisfied,
                            f"{ss.satisfied / total * 100:.1f}%",
                        )
                        col_sm2.metric(
                            "😐 Neutrales",
                            ss.neutral,
                            f"{ss.neutral / total * 100:.1f}%",
                        )
                        col_sm3.metric(
                            "😤 Frustrados",
                            ss.frustrated,
                            f"{ss.frustrated / total * 100:.1f}%",
                            delta_color="inverse",
                        )

                # Friction points
                if ss.top_friction_points:
                    st.subheader("Principales puntos de friccion")
                    for i, fp in enumerate(ss.top_friction_points):
                        with st.container():
                            col_fp1, col_fp2, col_fp3 = st.columns([4, 1, 1])
                            col_fp1.markdown(f"**{i+1}.** {fp.description}")
                            col_fp2.metric("Ocurrencias", fp.occurrences)
                            col_fp3.metric("% conversaciones", f"{fp.percentage:.1f}%")
                            st.progress(fp.percentage / 100)

            # ── Value Suggestions ──
            if results.suggestions:
                st.divider()
                st.subheader("Sugerencias de valor")

                category_map = {
                    "autogestion": ("🤖 Autogestion", "info"),
                    "conversion": ("📈 Conversion", "success"),
                    "cuello_botella": ("⚠️ Cuellos de botella", "warning"),
                    "quick_win": ("⚡ Quick Wins", "info"),
                }

                # Group by category
                grouped: dict[str, list] = {}
                for s in results.suggestions:
                    cat = s.category
                    if cat not in grouped:
                        grouped[cat] = []
                    grouped[cat].append(s)

                for cat, suggestions in grouped.items():
                    label, style = category_map.get(cat, (cat, "info"))
                    st.markdown(f"### {label}")

                    for s in suggestions:
                        impact_badge = {
                            "alto": "🔴",
                            "medio": "🟡",
                            "bajo": "🟢",
                        }.get(s.impact, "⚪")

                        with st.container(border=True):
                            col_sg1, col_sg2 = st.columns([4, 1])
                            col_sg1.markdown(f"**{s.title}**")
                            col_sg2.markdown(f"Impacto: {impact_badge} {s.impact}")
                            st.markdown(s.description)
                            if s.metric:
                                st.caption(f"📊 {s.metric}")
