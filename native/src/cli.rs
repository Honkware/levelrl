use crate::{Error, Result};
use std::collections::BTreeMap;
use std::env;
use std::fmt::Display;
use std::str::FromStr;

pub struct CliArgs {
    values: BTreeMap<String, Vec<String>>,
}

impl CliArgs {
    pub fn parse() -> Result<Self> {
        let args = env::args_os()
            .skip(1)
            .map(|arg| {
                arg.into_string()
                    .map_err(|_| Error::InvalidArgument("options must be valid UTF-8".to_string()))
            })
            .collect::<Result<Vec<_>>>()?;
        let mut values: BTreeMap<String, Vec<String>> = BTreeMap::new();
        let mut index = 0;
        while index < args.len() {
            let name = &args[index];
            if !name.starts_with("--") {
                return Err(Error::InvalidArgument(format!(
                    "unexpected positional argument `{name}`"
                )));
            }
            index += 1;
            let start = index;
            while index < args.len() && !args[index].starts_with("--") {
                index += 1;
            }
            if values
                .insert(name.clone(), args[start..index].to_vec())
                .is_some()
            {
                return Err(Error::InvalidArgument(format!("duplicate option `{name}`")));
            }
        }
        Ok(Self { values })
    }

    pub fn finish(self, allowed: &[&str]) -> Result<Self> {
        if let Some(name) = self
            .values
            .keys()
            .find(|name| !allowed.contains(&name.as_str()))
        {
            return Err(Error::InvalidArgument(format!("unknown option `{name}`")));
        }
        Ok(self)
    }

    pub fn value(&self, name: &str) -> Result<Option<&str>> {
        match self.values.get(name).map(Vec::as_slice) {
            None => Ok(None),
            Some([value]) => Ok(Some(value)),
            Some([]) => Err(Error::InvalidArgument(format!("{name} requires a value"))),
            Some(_) => Err(Error::InvalidArgument(format!(
                "{name} accepts exactly one value"
            ))),
        }
    }

    pub fn values(&self, name: &str) -> Result<Vec<String>> {
        let Some(values) = self.values.get(name) else {
            return Ok(Vec::new());
        };
        if values.is_empty() {
            return Err(Error::InvalidArgument(format!("{name} requires a value")));
        }
        Ok(values
            .iter()
            .flat_map(|value| value.split_whitespace())
            .map(str::to_string)
            .collect())
    }

    pub fn parse_value<T>(&self, name: &str, default: T) -> Result<T>
    where
        T: FromStr,
        T::Err: Display,
    {
        match self.value(name)? {
            Some(value) => value.parse().map_err(|error| {
                Error::InvalidArgument(format!(
                    "{name} expects a valid value, got `{value}`: {error}"
                ))
            }),
            None => Ok(default),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_values_are_errors() {
        let args = CliArgs {
            values: BTreeMap::from([("--seed".to_string(), Vec::new())]),
        };
        assert!(matches!(
            args.value("--seed"),
            Err(Error::InvalidArgument(_))
        ));
    }
}
