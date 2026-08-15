use indexmap::IndexMap;
use serde::Deserialize;
use std::io::Read;
use std::time::Duration;

use super::config::{Config, ConfigModelOverride, EnvKeys};
use super::model_providers::ModelProviderConfig;
use crate::sampling::ApiBackend;
use crate::sampling::types::{ReasoningEffort, ReasoningEffortOption};

pub(crate) const GROK_PROVIDER_ENV: &str = "GROK_PROVIDER";
const MAX_PROVIDER_CATALOG_BYTES: u64 = 16 * 1024 * 1024;
const MAX_PROVIDER_MODELS: usize = 5_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ProviderPreset {
    Glm,
    Kimi,
    Minimax,
    Openai,
    Openrouter,
    Longcat,
}

struct PresetValues {
    model: &'static str,
    base_url: &'static str,
    env_key: &'static str,
    name: &'static str,
    description: &'static str,
    api_backend: ApiBackend,
    context_window: u64,
    temperature: Option<f32>,
}

#[derive(Clone, Copy)]
struct ModelPreset {
    id: &'static str,
    name: &'static str,
    description: &'static str,
    context_window: u64,
    effort: Option<EffortProfile>,
}

#[derive(Clone, Copy)]
struct EffortProfile {
    default: ReasoningEffort,
    levels: &'static [ReasoningEffort],
}

const GLM_EFFORT: EffortProfile = EffortProfile {
    default: ReasoningEffort::Max,
    levels: &[
        ReasoningEffort::Max,
        ReasoningEffort::High,
        ReasoningEffort::Low,
    ],
};
const KIMI_EFFORT: EffortProfile = EffortProfile {
    default: ReasoningEffort::High,
    levels: &[
        ReasoningEffort::Max,
        ReasoningEffort::High,
        ReasoningEffort::Low,
    ],
};
const OPENAI_56_EFFORT: EffortProfile = EffortProfile {
    default: ReasoningEffort::Medium,
    levels: &[
        ReasoningEffort::None,
        ReasoningEffort::Low,
        ReasoningEffort::Medium,
        ReasoningEffort::High,
        ReasoningEffort::Xhigh,
        ReasoningEffort::Max,
    ],
};
const OPENAI_55_EFFORT: EffortProfile = EffortProfile {
    default: ReasoningEffort::Medium,
    levels: &[
        ReasoningEffort::None,
        ReasoningEffort::Low,
        ReasoningEffort::Medium,
        ReasoningEffort::High,
        ReasoningEffort::Xhigh,
    ],
};
const OPENAI_55_PRO_EFFORT: EffortProfile = EffortProfile {
    default: ReasoningEffort::High,
    levels: &[
        ReasoningEffort::Medium,
        ReasoningEffort::High,
        ReasoningEffort::Xhigh,
    ],
};
const OPENAI_54_EFFORT: EffortProfile = EffortProfile {
    default: ReasoningEffort::None,
    levels: &[
        ReasoningEffort::None,
        ReasoningEffort::Low,
        ReasoningEffort::Medium,
        ReasoningEffort::High,
        ReasoningEffort::Xhigh,
    ],
};
const OPENAI_54_PRO_EFFORT: EffortProfile = EffortProfile {
    default: ReasoningEffort::Medium,
    levels: &[
        ReasoningEffort::Medium,
        ReasoningEffort::High,
        ReasoningEffort::Xhigh,
    ],
};
const OPENAI_CODEX_EFFORT: EffortProfile = EffortProfile {
    default: ReasoningEffort::Medium,
    levels: &[
        ReasoningEffort::Low,
        ReasoningEffort::Medium,
        ReasoningEffort::High,
        ReasoningEffort::Xhigh,
    ],
};
const OPENAI_51_EFFORT: EffortProfile = EffortProfile {
    default: ReasoningEffort::None,
    levels: &[
        ReasoningEffort::None,
        ReasoningEffort::Low,
        ReasoningEffort::Medium,
        ReasoningEffort::High,
    ],
};
const OPENAI_5_EFFORT: EffortProfile = EffortProfile {
    default: ReasoningEffort::Medium,
    levels: &[
        ReasoningEffort::Minimal,
        ReasoningEffort::Low,
        ReasoningEffort::Medium,
        ReasoningEffort::High,
    ],
};
const OPENAI_O3_EFFORT: EffortProfile = EffortProfile {
    default: ReasoningEffort::Medium,
    levels: &[
        ReasoningEffort::Low,
        ReasoningEffort::Medium,
        ReasoningEffort::High,
    ],
};
const OPENAI_HIGH_ONLY_EFFORT: EffortProfile = EffortProfile {
    default: ReasoningEffort::High,
    levels: &[ReasoningEffort::High],
};

const GLM_MODELS: &[ModelPreset] = &[
    ModelPreset {
        id: "glm-5.3",
        name: "GLM 5.3",
        description: "Latest GLM flagship model for coding and long-horizon agent tasks",
        context_window: 1_000_000,
        effort: Some(GLM_EFFORT),
    },
    ModelPreset {
        id: "glm-5.2",
        name: "GLM 5.2",
        description: "GLM flagship model with a 1M-token context window",
        context_window: 1_000_000,
        effort: Some(GLM_EFFORT),
    },
    ModelPreset {
        id: "glm-5.1",
        name: "GLM 5.1",
        description: "GLM 5.1 agentic coding model",
        context_window: 200_000,
        effort: None,
    },
    ModelPreset {
        id: "glm-5-turbo",
        name: "GLM 5 Turbo",
        description: "Low-latency GLM 5 model for agentic coding",
        context_window: 200_000,
        effort: None,
    },
    ModelPreset {
        id: "glm-5",
        name: "GLM 5",
        description: "GLM 5 foundation model for agentic engineering",
        context_window: 200_000,
        effort: None,
    },
    ModelPreset {
        id: "glm-4.7",
        name: "GLM 4.7",
        description: "GLM coding and agent model",
        context_window: 200_000,
        effort: None,
    },
    ModelPreset {
        id: "glm-4.6",
        name: "GLM 4.6",
        description: "GLM coding model",
        context_window: 200_000,
        effort: None,
    },
    ModelPreset {
        id: "glm-4.5",
        name: "GLM 4.5",
        description: "GLM general-purpose model",
        context_window: 128_000,
        effort: None,
    },
    ModelPreset {
        id: "glm-4.5-air",
        name: "GLM 4.5 Air",
        description: "Fast GLM 4.5 model",
        context_window: 128_000,
        effort: None,
    },
];

const KIMI_MODELS: &[ModelPreset] = &[
    ModelPreset {
        id: "k3",
        name: "Kimi K3",
        description: "Kimi's flagship coding model with a 1M-token context window",
        context_window: 1_048_576,
        effort: Some(KIMI_EFFORT),
    },
    ModelPreset {
        id: "k3-256k",
        name: "Kimi K3 256K",
        description: "Kimi K3 with a quota-efficient 256K context window",
        context_window: 262_144,
        effort: Some(KIMI_EFFORT),
    },
    ModelPreset {
        id: "kimi-for-coding",
        name: "Kimi K2.7 Code",
        description: "Kimi coding model for routine development tasks",
        context_window: 262_144,
        effort: None,
    },
    ModelPreset {
        id: "kimi-for-coding-highspeed",
        name: "Kimi K2.7 Code HighSpeed",
        description: "High-throughput Kimi coding model",
        context_window: 262_144,
        effort: None,
    },
];

const MINIMAX_MODELS: &[ModelPreset] = &[
    ModelPreset {
        id: "MiniMax-M3",
        name: "MiniMax M3",
        description: "Latest MiniMax flagship model for coding and agent tasks",
        context_window: 1_000_000,
        effort: None,
    },
    ModelPreset {
        id: "MiniMax-M2.7",
        name: "MiniMax M2.7",
        description: "MiniMax model for software engineering and agent workflows",
        context_window: 204_800,
        effort: None,
    },
    ModelPreset {
        id: "MiniMax-M2.7-highspeed",
        name: "MiniMax M2.7 HighSpeed",
        description: "High-throughput MiniMax M2.7",
        context_window: 204_800,
        effort: None,
    },
    ModelPreset {
        id: "MiniMax-M2.5",
        name: "MiniMax M2.5",
        description: "MiniMax coding and reasoning model",
        context_window: 204_800,
        effort: None,
    },
    ModelPreset {
        id: "MiniMax-M2.5-highspeed",
        name: "MiniMax M2.5 HighSpeed",
        description: "High-throughput MiniMax M2.5",
        context_window: 204_800,
        effort: None,
    },
    ModelPreset {
        id: "MiniMax-M2.1",
        name: "MiniMax M2.1",
        description: "Legacy MiniMax coding model",
        context_window: 204_800,
        effort: None,
    },
    ModelPreset {
        id: "MiniMax-M2.1-highspeed",
        name: "MiniMax M2.1 HighSpeed",
        description: "High-throughput MiniMax M2.1",
        context_window: 204_800,
        effort: None,
    },
    ModelPreset {
        id: "MiniMax-M2",
        name: "MiniMax M2",
        description: "Legacy MiniMax agentic reasoning model",
        context_window: 204_800,
        effort: None,
    },
];

const OPENAI_MODELS: &[ModelPreset] = &[
    ModelPreset {
        id: "gpt-5.6-sol",
        name: "GPT-5.6 Sol",
        description: "OpenAI frontier model for complex professional work",
        context_window: 1_050_000,
        effort: Some(OPENAI_56_EFFORT),
    },
    ModelPreset {
        id: "gpt-5.6",
        name: "GPT-5.6",
        description: "Alias for OpenAI's GPT-5.6 Sol model",
        context_window: 1_050_000,
        effort: Some(OPENAI_56_EFFORT),
    },
    ModelPreset {
        id: "gpt-5.6-terra",
        name: "GPT-5.6 Terra",
        description: "OpenAI model balancing intelligence and cost",
        context_window: 1_050_000,
        effort: Some(OPENAI_56_EFFORT),
    },
    ModelPreset {
        id: "gpt-5.6-luna",
        name: "GPT-5.6 Luna",
        description: "OpenAI model optimized for cost-sensitive workloads",
        context_window: 1_050_000,
        effort: Some(OPENAI_56_EFFORT),
    },
    ModelPreset {
        id: "gpt-5.5",
        name: "GPT-5.5",
        description: "OpenAI frontier model for coding and professional work",
        context_window: 1_050_000,
        effort: Some(OPENAI_55_EFFORT),
    },
    ModelPreset {
        id: "gpt-5.5-pro",
        name: "GPT-5.5 Pro",
        description: "Higher-compute GPT-5.5 model",
        context_window: 1_050_000,
        effort: Some(OPENAI_55_PRO_EFFORT),
    },
    ModelPreset {
        id: "gpt-5.4",
        name: "GPT-5.4",
        description: "OpenAI model for coding and professional work",
        context_window: 1_050_000,
        effort: Some(OPENAI_54_EFFORT),
    },
    ModelPreset {
        id: "gpt-5.4-pro",
        name: "GPT-5.4 Pro",
        description: "Higher-compute GPT-5.4 model",
        context_window: 1_050_000,
        effort: Some(OPENAI_54_PRO_EFFORT),
    },
    ModelPreset {
        id: "gpt-5.4-mini",
        name: "GPT-5.4 mini",
        description: "Efficient OpenAI model for coding and subagents",
        context_window: 400_000,
        effort: Some(OPENAI_54_EFFORT),
    },
    ModelPreset {
        id: "gpt-5.4-nano",
        name: "GPT-5.4 nano",
        description: "Low-cost OpenAI model for high-volume tasks",
        context_window: 400_000,
        effort: Some(OPENAI_54_EFFORT),
    },
    ModelPreset {
        id: "gpt-5.3-codex",
        name: "GPT-5.3-Codex",
        description: "OpenAI model optimized for agentic coding",
        context_window: 400_000,
        effort: Some(OPENAI_CODEX_EFFORT),
    },
    ModelPreset {
        id: "gpt-5.2",
        name: "GPT-5.2",
        description: "OpenAI reasoning model for professional work",
        context_window: 400_000,
        effort: Some(OPENAI_54_EFFORT),
    },
    ModelPreset {
        id: "gpt-5.2-pro",
        name: "GPT-5.2 Pro",
        description: "Higher-compute GPT-5.2 model",
        context_window: 400_000,
        effort: Some(OPENAI_55_PRO_EFFORT),
    },
    ModelPreset {
        id: "gpt-5.1",
        name: "GPT-5.1",
        description: "OpenAI coding and agentic reasoning model",
        context_window: 400_000,
        effort: Some(OPENAI_51_EFFORT),
    },
    ModelPreset {
        id: "gpt-5",
        name: "GPT-5",
        description: "OpenAI general-purpose reasoning model",
        context_window: 400_000,
        effort: Some(OPENAI_5_EFFORT),
    },
    ModelPreset {
        id: "gpt-5-mini",
        name: "GPT-5 mini",
        description: "Cost-efficient GPT-5 model",
        context_window: 400_000,
        effort: Some(OPENAI_5_EFFORT),
    },
    ModelPreset {
        id: "gpt-5-nano",
        name: "GPT-5 nano",
        description: "Low-cost GPT-5 model",
        context_window: 400_000,
        effort: Some(OPENAI_5_EFFORT),
    },
    ModelPreset {
        id: "gpt-5-pro",
        name: "GPT-5 Pro",
        description: "Higher-compute GPT-5 model",
        context_window: 400_000,
        effort: Some(OPENAI_HIGH_ONLY_EFFORT),
    },
    ModelPreset {
        id: "o3-pro",
        name: "o3-pro",
        description: "Higher-compute OpenAI o3 reasoning model",
        context_window: 200_000,
        effort: Some(OPENAI_HIGH_ONLY_EFFORT),
    },
    ModelPreset {
        id: "o3",
        name: "o3",
        description: "OpenAI reasoning model for complex tasks",
        context_window: 200_000,
        effort: Some(OPENAI_O3_EFFORT),
    },
    ModelPreset {
        id: "gpt-4.1",
        name: "GPT-4.1",
        description: "OpenAI non-reasoning model",
        context_window: 1_047_576,
        effort: None,
    },
    ModelPreset {
        id: "gpt-4.1-mini",
        name: "GPT-4.1 mini",
        description: "Fast OpenAI non-reasoning model",
        context_window: 1_047_576,
        effort: None,
    },
    ModelPreset {
        id: "gpt-4o",
        name: "GPT-4o",
        description: "OpenAI multimodal general-purpose model",
        context_window: 128_000,
        effort: None,
    },
    ModelPreset {
        id: "gpt-4o-mini",
        name: "GPT-4o mini",
        description: "Fast, affordable OpenAI model",
        context_window: 128_000,
        effort: None,
    },
    ModelPreset {
        id: "chat-latest",
        name: "Chat Latest",
        description: "Latest OpenAI Chat model alias",
        context_window: 400_000,
        effort: None,
    },
];

const OPENROUTER_MODELS: &[ModelPreset] = &[ModelPreset {
    id: "openrouter/owl-alpha",
    name: "OpenRouter Auto",
    description: "OpenRouter's automatic model router",
    context_window: 1_050_000,
    effort: None,
}];

const LONGCAT_MODELS: &[ModelPreset] = &[ModelPreset {
    id: "LongCat-2.0",
    name: "LongCat 2.0",
    description: "LongCat model for coding and agentic tasks",
    context_window: 1_000_000,
    effort: None,
}];

#[derive(Debug, Deserialize)]
struct ProviderModelsResponse {
    #[serde(default)]
    data: Vec<DiscoveredModel>,
}

#[derive(Debug, Deserialize)]
struct DiscoveredModel {
    id: String,
    name: Option<String>,
    description: Option<String>,
    context_length: Option<u64>,
}

impl ProviderPreset {
    pub(crate) const ALL: [Self; 6] = [
        Self::Glm,
        Self::Kimi,
        Self::Minimax,
        Self::Openai,
        Self::Openrouter,
        Self::Longcat,
    ];

    pub(crate) fn from_name(name: &str) -> Option<Self> {
        match name.trim().to_ascii_lowercase().as_str() {
            "glm" | "zai" | "zhipu" => Some(Self::Glm),
            "kimi" => Some(Self::Kimi),
            "minimax" => Some(Self::Minimax),
            "openai" => Some(Self::Openai),
            "openrouter" => Some(Self::Openrouter),
            "longcat" => Some(Self::Longcat),
            _ => None,
        }
    }

    pub(crate) fn canonical_name(self) -> &'static str {
        match self {
            Self::Glm => "glm",
            Self::Kimi => "kimi",
            Self::Minimax => "minimax",
            Self::Openai => "openai",
            Self::Openrouter => "openrouter",
            Self::Longcat => "longcat",
        }
    }

    pub(crate) fn env_key(self) -> &'static str {
        self.values().env_key
    }

    pub(crate) fn has_env_credentials(self) -> bool {
        std::env::var(self.env_key())
            .ok()
            .is_some_and(|key| !key.trim().is_empty())
    }

    pub(crate) fn ensure_provider_defaults(
        self,
        providers: &mut IndexMap<String, ModelProviderConfig>,
    ) {
        let provider = providers
            .entry(self.canonical_name().to_owned())
            .or_default();
        self.fill_provider_defaults(provider);
    }

    pub(crate) fn install_model_defaults(
        models: &mut IndexMap<String, ConfigModelOverride>,
        providers: &mut IndexMap<String, ModelProviderConfig>,
    ) {
        for model in models.values_mut() {
            let Some(provider_name) = model.model_provider.as_deref() else {
                continue;
            };
            let Some(preset) = Self::from_name(provider_name) else {
                continue;
            };
            preset.ensure_provider_defaults(providers);
            preset.fill_model_defaults(model);
        }
    }

    pub(crate) fn install_selected_models(
        self,
        models: &mut IndexMap<String, ConfigModelOverride>,
        providers: &mut IndexMap<String, ModelProviderConfig>,
    ) -> String {
        self.ensure_provider_defaults(providers);
        for preset in self.model_catalog() {
            let model = models.entry(preset.id.to_owned()).or_default();
            model.model.get_or_insert_with(|| preset.id.to_owned());
            self.fill_model_defaults(model);
        }
        self.model_catalog()[0].id.to_owned()
    }

    pub(crate) fn fill_model_defaults(self, model: &mut ConfigModelOverride) {
        let values = self.values();
        let model_id = model.model.as_deref().unwrap_or(values.model);
        let catalog_model = self.catalog_model(model_id);
        let context_window = catalog_model
            .map(|preset| preset.context_window)
            .unwrap_or(values.context_window);
        let name = catalog_model
            .map(|preset| preset.name)
            .unwrap_or(values.name);
        let description = catalog_model
            .map(|preset| preset.description)
            .unwrap_or(values.description);

        model.model_provider = Some(self.canonical_name().to_owned());
        model.model.get_or_insert_with(|| values.model.to_owned());
        model.name.get_or_insert_with(|| name.to_owned());
        model
            .description
            .get_or_insert_with(|| description.to_owned());
        model.context_window.get_or_insert(context_window);
        if let Some(temperature) = values.temperature {
            model.temperature.get_or_insert(temperature);
        }
        model.supported_in_api.get_or_insert(true);
        if let Some(profile) = catalog_model.and_then(|preset| preset.effort) {
            Self::fill_reasoning_defaults(model, profile);
        }
    }

    fn fill_provider_defaults(self, provider: &mut ModelProviderConfig) {
        let values = self.values();
        provider
            .base_url
            .get_or_insert_with(|| values.base_url.to_owned());
        provider
            .env_key
            .get_or_insert_with(|| EnvKeys::single(values.env_key));
        provider
            .api_backend
            .get_or_insert(values.api_backend.clone());
        provider.context_window.get_or_insert(values.context_window);
        if self == Self::Openrouter {
            provider
                .extra_headers
                .entry("X-Title".to_owned())
                .or_insert_with(|| "Grok Build".to_owned());
        }
    }

    /// Refresh every enabled preset from its OpenAI-compatible model-list API.
    /// Static entries remain as offline fallbacks; discovery only adds models
    /// and fills metadata that the user did not override.
    pub(crate) fn refresh_catalogs(config: &mut Config) {
        if !crate::util::config::resolve_remote_fetch_enabled() {
            tracing::info!("provider model discovery skipped: remote_fetch disabled");
            return;
        }

        let jobs: Vec<_> = Self::ALL
            .into_iter()
            .filter(|preset| config.model_providers.contains_key(preset.canonical_name()))
            .filter_map(|preset| {
                let api_key = preset.discovery_api_key(&config.model_providers);
                (!preset.discovery_requires_auth() || api_key.is_some())
                    .then_some((preset, api_key))
            })
            .collect();

        let results = std::thread::scope(|scope| {
            let handles: Vec<_> = jobs
                .into_iter()
                .map(|(preset, api_key)| {
                    scope.spawn(move || (preset, preset.fetch_catalog(api_key.as_deref())))
                })
                .collect();
            handles
                .into_iter()
                .filter_map(|handle| handle.join().ok())
                .collect::<Vec<_>>()
        });

        for (preset, result) in results {
            match result {
                Ok(discovered) => {
                    let count = discovered.len();
                    preset.install_discovered_models(discovered, &mut config.config_models);
                    tracing::info!(
                        provider = preset.canonical_name(),
                        count,
                        "refreshed provider model catalog"
                    );
                }
                Err(error) => {
                    tracing::warn!(
                        provider = preset.canonical_name(),
                        %error,
                        "provider model discovery failed; using static catalog"
                    );
                }
            }
        }
    }

    fn discovery_api_key(
        self,
        providers: &IndexMap<String, ModelProviderConfig>,
    ) -> Option<String> {
        let provider = providers.get(self.canonical_name());
        provider
            .and_then(|config| {
                config
                    .api_key
                    .as_deref()
                    .filter(|key| !key.trim().is_empty())
                    .map(str::to_owned)
            })
            .or_else(|| {
                provider
                    .and_then(|config| config.env_key.as_ref())
                    .and_then(EnvKeys::resolve_value)
            })
            .or_else(|| {
                std::env::var(self.env_key())
                    .ok()
                    .filter(|key| !key.trim().is_empty())
            })
    }

    fn discovery_requires_auth(self) -> bool {
        self != Self::Openrouter
    }

    fn discovery_url(self) -> &'static str {
        match self {
            Self::Glm => "https://api.z.ai/api/coding/paas/v4/models",
            Self::Kimi => "https://api.kimi.com/coding/v1/models",
            Self::Minimax => "https://api.minimax.io/v1/models",
            Self::Openai => "https://api.openai.com/v1/models",
            Self::Openrouter => {
                "https://openrouter.ai/api/v1/models?output_modalities=text&supported_parameters=tools"
            }
            Self::Longcat => "https://api.longcat.chat/openai/v1/models",
        }
    }

    fn fetch_catalog(self, api_key: Option<&str>) -> Result<Vec<DiscoveredModel>, String> {
        let client = reqwest::blocking::Client::builder()
            .timeout(Duration::from_secs(5))
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .map_err(|error| error.to_string())?;
        let mut request = client.get(self.discovery_url());
        if let Some(api_key) = api_key {
            request = request.bearer_auth(api_key);
        }
        let response = request
            .send()
            .and_then(reqwest::blocking::Response::error_for_status)
            .map_err(|error| error.to_string())?;
        if response
            .content_length()
            .is_some_and(|length| length > MAX_PROVIDER_CATALOG_BYTES)
        {
            return Err("provider catalog exceeds the 16 MiB response limit".to_owned());
        }
        let mut bytes = Vec::new();
        response
            .take(MAX_PROVIDER_CATALOG_BYTES + 1)
            .read_to_end(&mut bytes)
            .map_err(|error| error.to_string())?;
        if bytes.len() as u64 > MAX_PROVIDER_CATALOG_BYTES {
            return Err("provider catalog exceeds the 16 MiB response limit".to_owned());
        }
        let mut models: ProviderModelsResponse =
            serde_json::from_slice(&bytes).map_err(|error| error.to_string())?;
        models.data.truncate(MAX_PROVIDER_MODELS);
        for model in &mut models.data {
            model.id = model.id.trim().to_owned();
            model.name = model
                .name
                .take()
                .and_then(|value| sanitize_catalog_text(value, 256));
            model.description = model
                .description
                .take()
                .and_then(|value| sanitize_catalog_text(value, 2_048));
            model.context_length = model
                .context_length
                .filter(|window| (1..=100_000_000).contains(window));
        }
        models.data.retain(|model| {
            is_safe_model_id(&model.id) && self.is_agent_compatible_model(&model.id)
        });
        Ok(models.data)
    }

    fn is_agent_compatible_model(self, id: &str) -> bool {
        if self != Self::Openai {
            return true;
        }

        let id = id.to_ascii_lowercase();
        let text_family = id.starts_with("gpt-")
            || id.starts_with("o1")
            || id.starts_with("o3")
            || id.starts_with("o4")
            || id.starts_with("ft:")
            || id == "chat-latest";
        let incompatible = [
            "audio",
            "realtime",
            "transcribe",
            "tts",
            "image",
            "search-preview",
            "deep-research",
            "computer-use",
        ]
        .into_iter()
        .any(|marker| id.contains(marker));
        text_family && !incompatible
    }

    fn install_discovered_models(
        self,
        discovered: Vec<DiscoveredModel>,
        models: &mut IndexMap<String, ConfigModelOverride>,
    ) {
        for discovered in discovered {
            let id = discovered.id.trim();
            let model = models.entry(id.to_owned()).or_default();
            model.model.get_or_insert_with(|| id.to_owned());
            if model.name.is_none() {
                model.name = discovered.name.filter(|name| !name.trim().is_empty());
            }
            if model.description.is_none() {
                model.description = discovered
                    .description
                    .filter(|description| !description.trim().is_empty());
            }
            if model.context_window.is_none() {
                model.context_window = discovered.context_length.filter(|window| *window > 0);
            }
            self.fill_model_defaults(model);
        }
    }

    fn model_catalog(self) -> &'static [ModelPreset] {
        match self {
            Self::Glm => GLM_MODELS,
            Self::Kimi => KIMI_MODELS,
            Self::Minimax => MINIMAX_MODELS,
            Self::Openai => OPENAI_MODELS,
            Self::Openrouter => OPENROUTER_MODELS,
            Self::Longcat => LONGCAT_MODELS,
        }
    }

    fn catalog_model(self, model: &str) -> Option<ModelPreset> {
        let normalized = model.trim().to_ascii_lowercase();
        let model_id = if self == Self::Glm {
            normalized.strip_suffix("[1m]").unwrap_or(&normalized)
        } else {
            &normalized
        };
        let exact = self
            .model_catalog()
            .iter()
            .copied()
            .find(|preset| preset.id.eq_ignore_ascii_case(model_id));
        if exact.is_some() || self != Self::Openai {
            return exact;
        }
        self.model_catalog()
            .iter()
            .copied()
            .filter(|preset| {
                model_id
                    .strip_prefix(preset.id)
                    .is_some_and(|suffix| suffix.starts_with('-'))
            })
            .max_by_key(|preset| preset.id.len())
    }

    fn fill_reasoning_defaults(model: &mut ConfigModelOverride, profile: EffortProfile) {
        if model.supports_reasoning_effort == Some(false) {
            return;
        }
        model.supports_reasoning_effort.get_or_insert(true);
        if !model.reasoning_efforts.is_empty() {
            return;
        }

        let default = *model.reasoning_effort.get_or_insert(profile.default);
        model.reasoning_efforts = profile
            .levels
            .iter()
            .copied()
            .map(|effort| ReasoningEffortOption {
                id: effort.as_str().to_owned(),
                value: effort,
                label: format!("{} Effort", effort_label(effort)),
                description: Some(effort_description(effort).to_owned()),
                default: effort == default,
            })
            .collect();
    }

    fn values(self) -> PresetValues {
        match self {
            Self::Glm => PresetValues {
                model: "glm-5.3",
                base_url: "https://api.z.ai/api/coding/paas/v4",
                env_key: "ZAI_API_KEY",
                name: "GLM 5.3 (Z.AI Coding Plan)",
                description: "Latest GLM 5.x coding model through the Z.AI Coding Plan endpoint",
                api_backend: ApiBackend::ChatCompletions,
                context_window: 1_000_000,
                temperature: Some(1.0),
            },
            Self::Kimi => PresetValues {
                model: "k3",
                base_url: "https://api.kimi.com/coding/v1",
                env_key: "KIMI_API_KEY",
                name: "Kimi K3",
                description: "Kimi K3 through Moonshot's coding endpoint",
                api_backend: ApiBackend::ChatCompletions,
                context_window: 1_048_576,
                temperature: None,
            },
            Self::Minimax => PresetValues {
                model: "MiniMax-M3",
                base_url: "https://api.minimax.io/v1",
                env_key: "MINIMAX_API_KEY",
                name: "MiniMax M3",
                description: "MiniMax M3 through the Token Plan endpoint",
                api_backend: ApiBackend::Responses,
                context_window: 1_000_000,
                temperature: None,
            },
            Self::Openai => PresetValues {
                model: "gpt-5.6-sol",
                base_url: "https://api.openai.com/v1",
                env_key: "OPENAI_API_KEY",
                name: "OpenAI",
                description: "OpenAI platform model through the Responses API",
                api_backend: ApiBackend::Responses,
                context_window: 1_050_000,
                temperature: None,
            },
            Self::Openrouter => PresetValues {
                model: "openrouter/owl-alpha",
                base_url: "https://openrouter.ai/api/v1",
                env_key: "OPENROUTER_API_KEY",
                name: "OpenRouter",
                description: "OpenRouter's OpenAI-compatible multi-model endpoint",
                api_backend: ApiBackend::ChatCompletions,
                context_window: 1_050_000,
                temperature: None,
            },
            Self::Longcat => PresetValues {
                model: "LongCat-2.0",
                base_url: "https://api.longcat.chat/openai/v1",
                env_key: "LONGCAT_API_KEY",
                name: "LongCat 2.0",
                description: "LongCat through Meituan's OpenAI-compatible endpoint",
                api_backend: ApiBackend::ChatCompletions,
                context_window: 1_000_000,
                temperature: Some(1.0),
            },
        }
    }
}

fn effort_label(effort: ReasoningEffort) -> &'static str {
    match effort {
        ReasoningEffort::None => "No",
        ReasoningEffort::Minimal => "Minimal",
        ReasoningEffort::Low => "Low",
        ReasoningEffort::Medium => "Medium",
        ReasoningEffort::High => "High",
        ReasoningEffort::Xhigh => "Extra High",
        ReasoningEffort::Max => "Max",
    }
}

fn effort_description(effort: ReasoningEffort) -> &'static str {
    match effort {
        ReasoningEffort::None => "Fastest response without additional reasoning",
        ReasoningEffort::Minimal => "Minimal reasoning for straightforward tasks",
        ReasoningEffort::Low => "Lightweight reasoning for faster turns",
        ReasoningEffort::Medium => "Balanced reasoning depth and latency",
        ReasoningEffort::High => "Strong reasoning for difficult tasks",
        ReasoningEffort::Xhigh => "Extra-deep reasoning for complex tasks",
        ReasoningEffort::Max => "Deepest available reasoning",
    }
}

fn is_safe_model_id(id: &str) -> bool {
    !id.is_empty()
        && id.len() <= 512
        && id
            .chars()
            .all(|character| !character.is_control() && !character.is_whitespace())
}

fn sanitize_catalog_text(value: String, max_chars: usize) -> Option<String> {
    let without_controls: String = value
        .chars()
        .filter(|character| !character.is_control() || character.is_whitespace())
        .collect();
    let sanitized: String = without_controls
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .chars()
        .take(max_chars)
        .collect();
    (!sanitized.is_empty()).then_some(sanitized)
}

#[cfg(test)]
mod tests {
    use serial_test::serial;
    use xai_grok_test_support::EnvGuard;

    use super::*;
    use crate::agent::config::{Config, resolve_credentials, resolve_model_list};
    use crate::auth::PreferredAuthMethod;
    use crate::sampling::types::ReasoningEffort;

    fn clear_provider_env() -> Vec<EnvGuard> {
        [
            GROK_PROVIDER_ENV,
            "ZAI_API_KEY",
            "KIMI_API_KEY",
            "MINIMAX_API_KEY",
            "OPENAI_API_KEY",
            "OPENROUTER_API_KEY",
            "LONGCAT_API_KEY",
        ]
        .into_iter()
        .map(EnvGuard::unset)
        .collect()
    }

    #[test]
    #[serial]
    fn glm_defaults_to_latest_model_and_coding_endpoint() {
        let _env = clear_provider_env();
        let raw: toml::Value = toml::from_str(
            r#"
            [model.glm]
            provider = "glm"
            "#,
        )
        .unwrap();

        let config = Config::new_from_toml_cfg(&raw).unwrap();
        let models = resolve_model_list(&config, None);
        let entry = models.get("glm").unwrap();
        assert_eq!(entry.info.model, "glm-5.3");
        assert_eq!(entry.info.base_url, "https://api.z.ai/api/coding/paas/v4");
        assert_eq!(entry.info.api_backend, ApiBackend::ChatCompletions);
        assert_eq!(entry.info.context_window.get(), 1_000_000);
        assert_eq!(entry.info.temperature, Some(1.0));
        assert_eq!(
            entry.env_key.as_ref().and_then(EnvKeys::primary),
            Some("ZAI_API_KEY")
        );
    }

    #[test]
    #[serial]
    fn glm_5x_overrides_use_their_documented_context_windows() {
        let _env = clear_provider_env();
        let raw: toml::Value = toml::from_str(
            r#"
            [model.glm-5]
            provider = "glm"
            model = "glm-5"

            [model.glm-5-1]
            provider = "glm"
            model = "glm-5.1"

            [model.glm-5-turbo]
            provider = "glm"
            model = "glm-5-turbo"

            [model.glm-5-2]
            provider = "glm"
            model = "glm-5.2"

            [model.glm-5-3]
            provider = "zai"
            model = "glm-5.3"
            "#,
        )
        .unwrap();

        let config = Config::new_from_toml_cfg(&raw).unwrap();
        let models = resolve_model_list(&config, None);
        for key in ["glm-5", "glm-5-1", "glm-5-turbo"] {
            assert_eq!(models[key].info.context_window.get(), 200_000, "{key}");
        }
        for key in ["glm-5-2", "glm-5-3"] {
            assert_eq!(models[key].info.context_window.get(), 1_000_000, "{key}");
        }
    }

    #[test]
    #[serial]
    fn explicit_fields_override_preset_defaults() {
        let _env = clear_provider_env();
        let raw: toml::Value = toml::from_str(
            r#"
            [model.custom]
            provider = "glm"
            model = "glm-private"
            base_url = "https://gateway.example/v1"
            env_key = "GATEWAY_API_KEY"
            context_window = 123456
            "#,
        )
        .unwrap();

        let config = Config::new_from_toml_cfg(&raw).unwrap();
        let models = resolve_model_list(&config, None);
        let entry = models.get("custom").unwrap();
        assert_eq!(entry.info.model, "glm-private");
        assert_eq!(entry.info.base_url, "https://gateway.example/v1");
        assert_eq!(entry.info.context_window.get(), 123_456);
        assert_eq!(
            entry.env_key.as_ref().and_then(EnvKeys::primary),
            Some("GATEWAY_API_KEY")
        );
    }

    #[test]
    #[serial]
    fn provider_env_selects_glm_and_uses_its_key() {
        let _env = clear_provider_env();
        let _provider = EnvGuard::set(GROK_PROVIDER_ENV, "zhipu");
        let _key = EnvGuard::set("ZAI_API_KEY", "test-zai-key");
        let raw: toml::Value = toml::from_str("").unwrap();

        let config = Config::new_from_toml_cfg(&raw).unwrap();
        assert_eq!(config.models.default.as_deref(), Some("glm-5.3"));
        assert_eq!(
            config.grok_com_config.preferred_method,
            Some(PreferredAuthMethod::ApiKey)
        );

        let models = resolve_model_list(&config, None);
        let entry = models.get("glm-5.3").unwrap();
        let credentials = resolve_credentials(entry, Some("must-not-be-used"));
        assert_eq!(credentials.api_key.as_deref(), Some("test-zai-key"));
        assert_eq!(entry.info.model, "glm-5.3");
    }

    #[test]
    #[serial]
    fn glm_selector_installs_versioned_catalog_with_native_effort_levels() {
        let _env = clear_provider_env();
        let _provider = EnvGuard::set(GROK_PROVIDER_ENV, "glm");
        let _key = EnvGuard::set("ZAI_API_KEY", "test-zai-key");
        let raw: toml::Value = toml::from_str("").unwrap();

        let config = Config::new_from_toml_cfg(&raw).unwrap();
        assert_eq!(config.models.default.as_deref(), Some("glm-5.3"));

        let models = resolve_model_list(&config, None);
        for model_id in ["glm-5.3", "glm-5.2", "glm-5.1", "glm-5-turbo", "glm-5"] {
            assert_eq!(models[model_id].info.model, model_id);
        }

        for model_id in ["glm-5.3", "glm-5.2"] {
            let info = &models[model_id].info;
            assert!(info.supports_reasoning_effort, "{model_id}");
            assert_eq!(info.reasoning_effort, Some(ReasoningEffort::Max));
            assert_eq!(
                info.reasoning_efforts
                    .iter()
                    .map(|option| option.value)
                    .collect::<Vec<_>>(),
                [
                    ReasoningEffort::Max,
                    ReasoningEffort::High,
                    ReasoningEffort::Low
                ]
            );
        }

        for model_id in ["glm-5.1", "glm-5-turbo", "glm-5"] {
            assert!(
                !models[model_id].info.supports_reasoning_effort,
                "{model_id}"
            );
        }
    }

    #[test]
    #[serial]
    fn explicit_glm_variant_gets_variant_name_and_efforts() {
        let _env = clear_provider_env();
        let raw: toml::Value = toml::from_str(
            r#"
            [model."glm-5.2"]
            provider = "glm"
            model = "glm-5.2"
            "#,
        )
        .unwrap();

        let config = Config::new_from_toml_cfg(&raw).unwrap();
        let models = resolve_model_list(&config, None);
        let info = &models["glm-5.2"].info;
        assert_eq!(info.name.as_deref(), Some("GLM 5.2"));
        assert_eq!(info.reasoning_effort, Some(ReasoningEffort::Max));
        assert!(info.supports_reasoning_effort);
    }

    #[test]
    #[serial]
    fn missing_glm_key_fails_closed_without_session_token_leakage() {
        let _env = clear_provider_env();
        let _provider = EnvGuard::set(GROK_PROVIDER_ENV, "glm");
        let raw: toml::Value = toml::from_str("").unwrap();

        let config = Config::new_from_toml_cfg(&raw).unwrap();
        let models = resolve_model_list(&config, None);
        let entry = models.get("glm-5.3").unwrap();
        assert!(
            entry
                .auth_provider
                .as_ref()
                .is_some_and(crate::auth::AuthProviderRef::is_fail_closed)
        );
        assert_eq!(
            resolve_credentials(entry, Some("xai-session-token")).api_key,
            None
        );
    }

    #[test]
    #[serial]
    fn configured_auto_catalogs_discovers_provider_keys_without_env_selector() {
        let _env = clear_provider_env();
        let _key = EnvGuard::set("ZAI_API_KEY", "test-zai-key");
        let raw: toml::Value = toml::from_str(
            r#"
            [models]
            provider_catalogs = ["auto"]
            "#,
        )
        .unwrap();

        let config = Config::new_from_toml_cfg(&raw).unwrap();
        assert_eq!(config.models.default.as_deref(), Some("glm-5.3"));
        let models = resolve_model_list(&config, None);
        assert!(models.contains_key("glm-5.3"));
        assert!(models.contains_key("glm-5.2"));
    }

    #[test]
    #[serial]
    fn auto_selector_discovers_configured_provider_keys() {
        let _env = clear_provider_env();
        let _provider = EnvGuard::set(GROK_PROVIDER_ENV, "auto");
        let _glm = EnvGuard::set("ZAI_API_KEY", "test-zai-key");
        let _minimax = EnvGuard::set("MINIMAX_API_KEY", "test-minimax-key");
        let raw: toml::Value = toml::from_str("").unwrap();

        let config = Config::new_from_toml_cfg(&raw).unwrap();
        assert_eq!(config.models.default.as_deref(), Some("glm-5.3"));
        let models = resolve_model_list(&config, None);
        assert!(models.contains_key("glm-5.3"));
        assert!(models.contains_key("glm-5.2"));
        assert!(models.contains_key("MiniMax-M3"));
        assert!(models.contains_key("MiniMax-M2.7"));
        assert!(!models.contains_key("k3"));
    }

    #[test]
    #[serial]
    fn all_selector_installs_complete_agent_model_catalogs() {
        let _env = clear_provider_env();
        let _provider = EnvGuard::set(GROK_PROVIDER_ENV, "all");
        let raw: toml::Value = toml::from_str("").unwrap();

        let config = Config::new_from_toml_cfg(&raw).unwrap();
        let models = resolve_model_list(&config, None);

        for model_id in [
            "glm-5.3",
            "glm-4.7",
            "k3",
            "k3-256k",
            "kimi-for-coding",
            "kimi-for-coding-highspeed",
            "MiniMax-M3",
            "MiniMax-M2.7",
            "MiniMax-M2.7-highspeed",
            "gpt-5.6-sol",
            "gpt-5.6-terra",
            "gpt-5.6-luna",
            "gpt-5.5",
            "gpt-5.4-mini",
            "gpt-5.3-codex",
            "openrouter/owl-alpha",
            "LongCat-2.0",
        ] {
            assert!(models.contains_key(model_id), "missing {model_id}");
        }

        let kimi = &models["k3"].info;
        assert_eq!(kimi.context_window.get(), 1_048_576);
        assert_eq!(kimi.reasoning_effort, Some(ReasoningEffort::High));
        assert_eq!(
            kimi.reasoning_efforts
                .iter()
                .map(|option| option.value)
                .collect::<Vec<_>>(),
            [
                ReasoningEffort::Max,
                ReasoningEffort::High,
                ReasoningEffort::Low
            ]
        );

        let openai = &models["gpt-5.6-sol"].info;
        assert_eq!(openai.context_window.get(), 1_050_000);
        assert_eq!(openai.reasoning_effort, Some(ReasoningEffort::Medium));
        assert_eq!(
            openai
                .reasoning_efforts
                .iter()
                .map(|option| option.value)
                .collect::<Vec<_>>(),
            [
                ReasoningEffort::None,
                ReasoningEffort::Low,
                ReasoningEffort::Medium,
                ReasoningEffort::High,
                ReasoningEffort::Xhigh,
                ReasoningEffort::Max,
            ]
        );
    }

    #[test]
    #[serial]
    fn config_can_persist_all_provider_catalogs() {
        let _env = clear_provider_env();
        let raw: toml::Value = toml::from_str(
            r#"
            [models]
            default = "glm-5.3"
            provider_catalogs = ["all"]
            "#,
        )
        .unwrap();

        let config = Config::new_from_toml_cfg(&raw).unwrap();
        let models = resolve_model_list(&config, None);
        for model_id in [
            "glm-5.3",
            "k3",
            "MiniMax-M3",
            "gpt-5.6-sol",
            "openrouter/owl-alpha",
            "LongCat-2.0",
        ] {
            assert!(models.contains_key(model_id), "missing {model_id}");
        }
    }

    #[test]
    fn discovered_entries_preserve_provider_routing_and_remote_metadata() {
        let response: ProviderModelsResponse = serde_json::from_value(serde_json::json!({
            "data": [{
                "id": "vendor/new-agent",
                "name": "New Agent",
                "description": "A newly released tool-capable model",
                "context_length": 765432
            }]
        }))
        .unwrap();
        let mut models = IndexMap::new();
        let mut providers = IndexMap::new();
        ProviderPreset::Openrouter.install_selected_models(&mut models, &mut providers);
        ProviderPreset::Openrouter.install_discovered_models(response.data, &mut models);

        let discovered = &models["vendor/new-agent"];
        assert_eq!(discovered.model.as_deref(), Some("vendor/new-agent"));
        assert_eq!(discovered.model_provider.as_deref(), Some("openrouter"));
        assert_eq!(discovered.name.as_deref(), Some("New Agent"));
        assert_eq!(discovered.context_window, Some(765_432));
    }

    #[test]
    fn openai_discovery_filters_non_agent_api_models() {
        let preset = ProviderPreset::Openai;
        assert!(preset.is_agent_compatible_model("gpt-5.6-sol"));
        assert!(preset.is_agent_compatible_model("ft:gpt-5.4:org:agent"));
        assert!(!preset.is_agent_compatible_model("gpt-image-2"));
        assert!(!preset.is_agent_compatible_model("gpt-realtime-2.1"));
        assert!(!preset.is_agent_compatible_model("text-embedding-3-large"));
    }

    #[test]
    fn remote_catalog_fields_are_safe_for_terminal_display() {
        assert!(is_safe_model_id("vendor/agent:model-v1"));
        assert!(!is_safe_model_id("vendor/bad model"));
        assert!(!is_safe_model_id("vendor/bad\nmodel"));
        assert!(!is_safe_model_id(&"x".repeat(513)));

        assert_eq!(
            sanitize_catalog_text("  New\tAgent\nModel\u{1b}\u{7}  ".to_owned(), 64),
            Some("New Agent Model".to_owned())
        );
        assert_eq!(
            sanitize_catalog_text("long catalog name".to_owned(), 4),
            Some("long".to_owned())
        );
        assert_eq!(sanitize_catalog_text("\u{1b}\u{7}".to_owned(), 64), None);
    }

    #[test]
    #[serial]
    fn openai_snapshots_inherit_family_context_and_effort_levels() {
        let _env = clear_provider_env();
        let raw: toml::Value = toml::from_str(
            r#"
            [model."gpt-5.6-sol-2026-08-01"]
            provider = "openai"
            model = "gpt-5.6-sol-2026-08-01"
            "#,
        )
        .unwrap();

        let config = Config::new_from_toml_cfg(&raw).unwrap();
        let models = resolve_model_list(&config, None);
        let info = &models["gpt-5.6-sol-2026-08-01"].info;
        assert_eq!(info.context_window.get(), 1_050_000);
        assert_eq!(info.reasoning_effort, Some(ReasoningEffort::Medium));
        assert!(
            info.reasoning_efforts
                .iter()
                .any(|option| option.value == ReasoningEffort::Max)
        );
    }

    #[test]
    #[serial]
    fn invalid_provider_env_fails_before_authentication() {
        let _env = clear_provider_env();
        let _provider = EnvGuard::set(GROK_PROVIDER_ENV, "unknown");
        let raw: toml::Value = toml::from_str("").unwrap();

        let error = Config::new_from_toml_cfg(&raw).unwrap_err();
        assert!(error.contains("unsupported GROK_PROVIDER value"), "{error}");
    }
}
